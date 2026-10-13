# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Model flexibility: swap the embedding model
# MAGIC
# MAGIC **The question:** *"Can I choose the embedding model and the LLM, or is it abstracted away?"*
# MAGIC
# MAGIC In this architecture the model is just a parameter. This notebook re-embeds a subset of articles with a
# MAGIC **second** Databricks-hosted embedding model (`alt_embedding_endpoint`), stores the vectors in a parallel
# MAGIC table, and compares top-5 results. The storage, indexing, and SQL stay the same.
# MAGIC
# MAGIC **Bringing your own model.** To keep Azure OpenAI `text-embedding-3-small`, register it as an
# MAGIC [external model endpoint](https://docs.databricks.com/en/generative-ai/external-models/index.html) in AI Gateway:
# MAGIC ```
# MAGIC databricks serving-endpoints create --json '{
# MAGIC   "name": "azure-oai-embeddings",
# MAGIC   "config": {"served_entities": [{
# MAGIC     "external_model": {
# MAGIC       "name": "text-embedding-3-small", "provider": "openai", "task": "llm/v1/embeddings",
# MAGIC       "openai_config": {
# MAGIC         "openai_api_type": "azure",
# MAGIC         "openai_api_key": "{{secrets/rag_demo/azure_oai_key}}",
# MAGIC         "openai_api_base": "https://<resource>.openai.azure.com/",
# MAGIC         "openai_deployment_name": "text-embedding-3-small",
# MAGIC         "openai_api_version": "2024-02-01"}}}]}}'
# MAGIC ```
# MAGIC Then set `alt_embedding_endpoint=azure-oai-embeddings`. The code below doesn't change, and AI Gateway adds
# MAGIC rate limits, usage tracking, and payload logging on top.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

import pandas as pd
from databricks.sdk import WorkspaceClient

from rag_common import get_params, pg_connect, vec_literal, embed

p = get_params(dbutils)
w = WorkspaceClient()
CAT, SCH = p["catalog"], p["schema"]
EMB, ALT = p["embedding_endpoint"], p["alt_embedding_endpoint"]
SUBSET = 500  # articles

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.{SCH}.chunk_embeddings_alt AS
SELECT chunk_id, ai_query('{ALT}', concat(title, '\\n\\n', chunk_text)) AS embedding
FROM {CAT}.{SCH}.chunk_embeddings
WHERE article_id IN (SELECT article_id FROM {CAT}.{SCH}.articles ORDER BY article_id LIMIT {SUBSET})
""")
alt_rows = spark.table(f"{CAT}.{SCH}.chunk_embeddings_alt").collect()
ALT_DIM = len(alt_rows[0].embedding)
print(f"{ALT}: {ALT_DIM} dims, {len(alt_rows)} chunks")

# COMMAND ----------

with pg_connect(p, w) as conn:
    conn.execute(f"""
      DROP TABLE IF EXISTS content.chunks_alt;
      CREATE TABLE content.chunks_alt (
        chunk_id  BIGINT PRIMARY KEY REFERENCES content.chunks (chunk_id) ON DELETE CASCADE,
        embedding VECTOR({ALT_DIM})
      )""")
    with conn.cursor() as cur, cur.copy("COPY content.chunks_alt (chunk_id, embedding) FROM STDIN") as cp:
        for r in alt_rows:
            cp.write_row((r.chunk_id, vec_literal(r.embedding)))
    conn.execute("CREATE INDEX ON content.chunks_alt USING hnsw (embedding vector_cosine_ops)")
    conn.execute("ANALYZE content.chunks_alt")

# COMMAND ----------

# MAGIC %md ## Same queries, two models (restricted to the same 500-article subset)

# COMMAND ----------

QUERIES = [
    "EV charging infrastructure for depot fleets",
    "school bus driver shortage",
    "telematics ROI for small fleets",
    "how do I get utility money to pay for chargers",
]

conn = pg_connect(p, w)
out = []
for q in QUERIES:
    e1, e2 = embed(w, EMB, [q])[0], embed(w, ALT, [q])[0]
    a = conn.execute(
        """SELECT c.title FROM content.chunks c JOIN content.chunks_alt x USING (chunk_id)
           ORDER BY c.embedding <=> %s::vector LIMIT 5""", (vec_literal(e1),)).fetchall()
    b = conn.execute(
        """SELECT c.title FROM content.chunks_alt x JOIN content.chunks c USING (chunk_id)
           ORDER BY x.embedding <=> %s::vector LIMIT 5""", (vec_literal(e2),)).fetchall()
    overlap = len({t for (t,) in a} & {t for (t,) in b})
    for i in range(5):
        out.append({"query": q, "rank": i + 1, EMB: a[i][0] if i < len(a) else None,
                    ALT: b[i][0] if i < len(b) else None, "top5_overlap": overlap})
conn.close()
display(pd.DataFrame(out))
