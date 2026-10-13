# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Hybrid search: vector + keyword in one Postgres query
# MAGIC
# MAGIC `content.search_chunks()` runs a pgvector HNSW search and a Postgres full-text search, then fuses
# MAGIC the two ranked lists with **Reciprocal Rank Fusion**. Filters (brand, tags, type, dates) are plain SQL `WHERE` clauses.
# MAGIC
# MAGIC Below, each query is compared across **vector only**, **keyword only** and **hybrid**, followed by a latency benchmark.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

import json
from datetime import datetime, timezone

import pandas as pd
from databricks.sdk import WorkspaceClient

from rag_common import get_params, pg_connect, vec_literal, embed, Timer

p = get_params(dbutils)
w = WorkspaceClient()
EMB = p["embedding_endpoint"]

QUERIES = [
    "EV charging infrastructure for depot fleets",
    "school bus driver shortage",
    "telematics ROI for small fleets",
    "nuclear verdicts insurance",                 # short jargon: keyword search helps
    "how do I get utility money to pay for chargers",  # paraphrase: vector search helps
    "ADAS calibration after windshield replacement",
]

SQL = """
SELECT article_id, brand, title, published_at, vector_rank, keyword_rank, round(score::numeric, 4) AS score
FROM content.search_chunks(%(q)s, %(e)s::vector, k => 8, mode => %(mode)s)
"""

conn = pg_connect(p, w)

def run(q, mode, emb=None):
    emb = emb or embed(w, EMB, [q])[0]
    rows = conn.execute(SQL, {"q": q, "e": vec_literal(emb), "mode": mode}).fetchall()
    return pd.DataFrame(rows, columns=["article_id", "brand", "title", "published_at", "vector_rank", "keyword_rank", "score"])

# COMMAND ----------

for q in QUERIES:
    emb = embed(w, EMB, [q])[0]
    print("=" * 100, f"\nQUERY: {q}")
    for mode in ["vector", "keyword", "hybrid"]:
        df = run(q, mode, emb)
        print(f"\n--- {mode} ({len(df)} hits)")
        print(df[["brand", "title", "published_at"]].head(5).to_string(index=False) if len(df) else "(no hits)")

# COMMAND ----------

# MAGIC %md ## Filters are just SQL

# COMMAND ----------

q = "EV charging infrastructure for depot fleets"
emb = embed(w, EMB, [q])[0]
rows = conn.execute(
    """SELECT brand, title, published_at, tags FROM content.search_chunks(
         %s, %s::vector, k => 8, brand_filter => ARRAY['Transit Forward'], date_from => DATE '2024-01-01')""",
    (q, vec_literal(emb)),
).fetchall()
display(pd.DataFrame(rows, columns=["brand", "title", "published_at", "tags"]))

# COMMAND ----------

# MAGIC %md ## Query plan: confirm HNSW + GIN indexes are used

# COMMAND ----------

for line in conn.execute(
    "EXPLAIN SELECT chunk_id FROM content.chunks ORDER BY embedding <=> %s::vector LIMIT 60", (vec_literal(emb),)
).fetchall():
    print(line[0])
print()
for line in conn.execute(
    "EXPLAIN SELECT chunk_id FROM content.chunks WHERE tsv @@ websearch_to_tsquery('english', %s)", (q,)
).fetchall():
    print(line[0])

# COMMAND ----------

# MAGIC %md ## Latency benchmark
# MAGIC Embedding latency (model serving) and retrieval latency (Lakebase) are measured separately.

# COMMAND ----------

bench_q = QUERIES * 17  # ~100 queries
embed_ms, db_ms = [], []
for q in bench_q:
    with Timer() as te:
        e = embed(w, EMB, [q])[0]
    with Timer() as td:
        conn.execute(SQL, {"q": q, "e": vec_literal(e), "mode": "hybrid"}).fetchall()
    embed_ms.append(te.ms)
    db_ms.append(td.ms)

def pct(xs, k):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * k))], 1)

result = {
    "n": len(bench_q),
    "hybrid_retrieval_ms_p50": pct(db_ms, 0.5),
    "hybrid_retrieval_ms_p95": pct(db_ms, 0.95),
    "embed_ms_p50": pct(embed_ms, 0.5),
    "embed_ms_p95": pct(embed_ms, 0.95),
}
print(json.dumps(result, indent=2))
conn.close()

spark.createDataFrame([(datetime.now(timezone.utc).isoformat(), "hybrid_search_bench", json.dumps(result))],
                      "run_at string, step string, metrics string") \
     .write.mode("append").saveAsTable(f"{p['catalog']}.{p['schema']}.run_metrics")
