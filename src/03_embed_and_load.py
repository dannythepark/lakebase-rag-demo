# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Chunk → embed → load into Lakebase → index
# MAGIC
# MAGIC * **Chunking:** split on paragraphs and pack to about 350 words (about 500 tokens), with about 50 words of overlap
# MAGIC * **Embedding:** `ai_query` against a Databricks-hosted embedding endpoint (configurable with the `embedding_endpoint` parameter)
# MAGIC * **Load:** Postgres `COPY` into Lakebase
# MAGIC * **Index:** HNSW (cosine) on the vector column and GIN on the full-text column, built **after** the load
# MAGIC
# MAGIC Run metrics (rows, tokens, timings, storage) go to the `run_metrics` table and feed the cost model.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

import json
import time
from datetime import datetime, timezone

from pyspark.sql import functions as F, types as T

from rag_common import get_params, pg_connect, vec_literal, Timer

p = get_params(dbutils)
CAT, SCH = p["catalog"], p["schema"]
EMB = p["embedding_endpoint"]
metrics = {"run_at": datetime.now(timezone.utc).isoformat(), "embedding_endpoint": EMB}

# COMMAND ----------

# MAGIC %md ## Chunk

# COMMAND ----------

TARGET_WORDS, OVERLAP_WORDS = 350, 50

@F.udf(T.ArrayType(T.StringType()))
def chunk_text(body):
    if not body:
        return []
    paras = [x.strip() for x in body.split("\n") if x.strip()]
    chunks, cur = [], []
    for para in paras:
        words = para.split()
        if cur and len(cur) + len(words) > TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP_WORDS:]
        cur.extend(words)
    if cur and (not chunks or len(cur) > OVERLAP_WORDS):
        chunks.append(" ".join(cur))
    return chunks

articles = spark.table(f"{CAT}.{SCH}.articles")
chunks = (
    articles.select("*", F.posexplode(chunk_text("body")).alias("chunk_index", "chunk_text"))
    .drop("body")
)
# stable ids: article number * 100 + chunk index
chunks = chunks.withColumn(
    "chunk_id", (F.regexp_extract("article_id", r"(\d+)", 1).cast("bigint") * 100 + F.col("chunk_index")).cast("bigint")
)
chunks.createOrReplaceTempView("chunks_v")
print("chunks:", chunks.count(), "from articles:", articles.count())

# COMMAND ----------

# MAGIC %md ## Embed with `ai_query`
# MAGIC The title is prepended to each chunk so short chunks still carry their headline's meaning.

# COMMAND ----------

with Timer() as t:
    spark.sql(f"""
    CREATE OR REPLACE TABLE {CAT}.{SCH}.chunk_embeddings AS
    SELECT chunk_id, article_id, chunk_index, brand, title, url, content_type, published_at, tags, author, chunk_text,
           ai_query('{EMB}', concat(title, '\\n\\n', chunk_text)) AS embedding
    FROM chunks_v
    """)
    n_chunks = spark.table(f"{CAT}.{SCH}.chunk_embeddings").count()
metrics["chunks"] = n_chunks
metrics["chunk_embed_seconds"] = round(t.ms / 1000, 1)
# rough token estimate: ~1.33 tokens per word
metrics["chunk_embed_tokens_est"] = int(
    spark.sql(f"SELECT sum(size(split(concat(title,' ',chunk_text),' '))) FROM {CAT}.{SCH}.chunk_embeddings").first()[0] * 1.33
)

with Timer() as t:
    spark.sql(f"""
    CREATE OR REPLACE TABLE {CAT}.{SCH}.campaign_embeddings AS
    SELECT *, ai_query('{EMB}', concat(name, '. ', package_type, '. ', description, ' Audience: ', target_audience)) AS embedding
    FROM {CAT}.{SCH}.campaigns
    """)
metrics["campaigns"] = spark.table(f"{CAT}.{SCH}.campaign_embeddings").count()
metrics["campaign_embed_seconds"] = round(t.ms / 1000, 1)

dim = len(spark.table(f"{CAT}.{SCH}.chunk_embeddings").select("embedding").first()[0])
assert dim == int(p["embedding_dim"]), f"model returns {dim} dims; set embedding_dim={dim} and rerun 02"
metrics["embedding_dim"] = dim
print(metrics)

# COMMAND ----------

# MAGIC %md ## Load into Lakebase with `COPY`

# COMMAND ----------

chunk_rows = spark.table(f"{CAT}.{SCH}.chunk_embeddings").toLocalIterator()
camp_rows = spark.table(f"{CAT}.{SCH}.campaign_embeddings").collect()

with pg_connect(p) as conn, Timer() as t:
    conn.execute("DROP INDEX IF EXISTS content.chunks_embedding_hnsw, content.chunks_tsv_gin, "
                 "content.campaigns_embedding_hnsw, content.campaigns_tsv_gin")
    conn.execute("TRUNCATE content.chunks, content.campaigns CASCADE")  # also clears chunks_alt; 05 rebuilds it
    with conn.cursor() as cur:
        with cur.copy(
            "COPY content.chunks (chunk_id, article_id, chunk_index, brand, title, url, content_type, "
            "published_at, tags, author, chunk_text, embedding) FROM STDIN"
        ) as cp:
            for r in chunk_rows:
                cp.write_row((r.chunk_id, r.article_id, r.chunk_index, r.brand, r.title, r.url, r.content_type,
                              r.published_at, list(r.tags or []), r.author, r.chunk_text, vec_literal(r.embedding)))
        with cur.copy(
            "COPY content.campaigns (campaign_id, record_type, name, package_type, sponsor_category, brands, "
            "target_audience, price_tier, price_range_usd, year, description, embedding) FROM STDIN"
        ) as cp:
            for r in camp_rows:
                cp.write_row((r.campaign_id, r.record_type, r.name, r.package_type, r.sponsor_category,
                              list(r.brands or []), r.target_audience, r.price_tier, r.price_range_usd, r.year,
                              r.description, vec_literal(r.embedding)))
metrics["copy_seconds"] = round(t.ms / 1000, 1)

# COMMAND ----------

# MAGIC %md ## Build indexes (after the load, so HNSW is built over real data)

# COMMAND ----------

with pg_connect(p) as conn, Timer() as t:
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute("CREATE INDEX chunks_embedding_hnsw ON content.chunks USING hnsw (embedding vector_cosine_ops) "
                 "WITH (m = 16, ef_construction = 64)")
    conn.execute("CREATE INDEX chunks_tsv_gin ON content.chunks USING gin (tsv)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_brand_date ON content.chunks (brand, published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tags_gin ON content.chunks USING gin (tags)")
    conn.execute("CREATE INDEX campaigns_embedding_hnsw ON content.campaigns USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX campaigns_tsv_gin ON content.campaigns USING gin (tsv)")
    conn.execute("ANALYZE content.chunks")
    conn.execute("ANALYZE content.campaigns")
metrics["index_seconds"] = round(t.ms / 1000, 1)

with pg_connect(p) as conn:
    sizes = conn.execute("""
      SELECT relname, pg_total_relation_size(c.oid)
      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'content' AND c.relkind IN ('r', 'i')
    """).fetchall()
    metrics["pg_bytes_by_relation"] = {k: v for k, v in sizes}
    metrics["pg_total_bytes"] = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    print(conn.execute("SELECT count(*), count(DISTINCT article_id) FROM content.chunks").fetchone())

print(json.dumps(metrics, indent=2, default=str))

# COMMAND ----------

spark.createDataFrame([(metrics["run_at"], "embed_and_load", json.dumps(metrics, default=str))],
                      "run_at string, step string, metrics string") \
     .write.mode("append").saveAsTable(f"{CAT}.{SCH}.run_metrics")
