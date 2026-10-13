# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Lakebase project, pgvector, schema
# MAGIC
# MAGIC Everything below runs in code, with nothing done by hand in the UI:
# MAGIC 1. Create the Lakebase Autoscaling project if it doesn't exist (scale-to-zero on, 0.5–4 CU)
# MAGIC 2. Enable `pgvector`
# MAGIC 3. Create the `content` schema: chunk tables with a vector column **and** a generated full-text `tsvector` column
# MAGIC 4. Create the hybrid search SQL functions (vector + keyword, fused with Reciprocal Rank Fusion)
# MAGIC
# MAGIC One Postgres database holds transactional data, vectors and full-text search, so no separate search service is needed.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from rag_common import get_params, pg_connect, endpoint_path, Lakebase, load_pgvector

p = get_params(dbutils)
w = WorkspaceClient()
lb = Lakebase(w)
DIM = int(p["embedding_dim"])
project_name = f"projects/{p['lakebase_project']}"

# COMMAND ----------

# MAGIC %md ## 1 · Create the project (idempotent)

# COMMAND ----------

if not lb.exists(project_name):
    print(f"Creating {project_name} ...")
    lb.create("projects", "project_id", p["lakebase_project"], {
        "spec": {
            "display_name": "Lakebase RAG Demo (Lakebase + pgvector)",
            "default_endpoint_settings": {
                "autoscaling_limit_min_cu": 0.5,
                "autoscaling_limit_max_cu": 4,
                "suspend_timeout_duration": "300s",  # scale to zero after 5 idle minutes
            },
        }
    })
else:
    print(f"{project_name} already exists")

ep = lb.get(endpoint_path(p))["status"]
print("host:", ep["hosts"]["host"])
print("autoscaling CU:", ep["autoscaling_limit_min_cu"], "-", ep["autoscaling_limit_max_cu"])
print("suspend after idle:", ep.get("suspend_timeout_duration"))

# COMMAND ----------

# MAGIC %md ## 2–4 · Extension, tables, search functions

# COMMAND ----------

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS content;

-- Editorial content: one row per chunk of an article
CREATE TABLE IF NOT EXISTS content.chunks (
  chunk_id      BIGINT PRIMARY KEY,
  article_id    TEXT NOT NULL,
  chunk_index   INT  NOT NULL,
  brand         TEXT NOT NULL,
  title         TEXT NOT NULL,
  url           TEXT,
  content_type  TEXT,
  published_at  DATE,
  tags          TEXT[],
  author        TEXT,
  chunk_text    TEXT NOT NULL,
  embedding     VECTOR({DIM}),
  tsv           TSVECTOR GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(chunk_text, '')), 'B')
                ) STORED
);

-- Media kit offerings + past campaign case studies (one chunk per record)
CREATE TABLE IF NOT EXISTS content.campaigns (
  campaign_id      TEXT PRIMARY KEY,
  record_type      TEXT NOT NULL,
  name             TEXT NOT NULL,
  package_type     TEXT,
  sponsor_category TEXT,
  brands           TEXT[],
  target_audience  TEXT,
  price_tier       TEXT,
  price_range_usd  TEXT,
  year             INT,
  description      TEXT NOT NULL,
  embedding        VECTOR({DIM}),
  tsv              TSVECTOR GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', coalesce(name, '') || ' ' || coalesce(package_type, '')), 'A') ||
                     setweight(to_tsvector('english', coalesce(description, '') || ' ' || coalesce(target_audience, '')), 'B')
                   ) STORED
);

-- Hybrid search over editorial chunks. mode: 'hybrid' | 'vector' | 'keyword'
CREATE OR REPLACE FUNCTION content.search_chunks(
  q_text       TEXT,
  q_emb        VECTOR({DIM}),
  k            INT     DEFAULT 20,
  mode         TEXT    DEFAULT 'hybrid',
  brand_filter TEXT[]  DEFAULT NULL,
  tag_filter   TEXT[]  DEFAULT NULL,
  type_filter  TEXT[]  DEFAULT NULL,
  date_from    DATE    DEFAULT NULL,
  date_to      DATE    DEFAULT NULL,
  candidates   INT     DEFAULT 60,
  rrf_k        INT     DEFAULT 60
)
RETURNS TABLE (
  chunk_id BIGINT, article_id TEXT, chunk_index INT, brand TEXT, title TEXT, url TEXT,
  content_type TEXT, published_at DATE, tags TEXT[], chunk_text TEXT,
  vector_rank BIGINT, keyword_rank BIGINT, cosine_sim FLOAT8, score FLOAT8
)
LANGUAGE plpgsql STABLE
SET hnsw.iterative_scan = 'relaxed_order'
SET hnsw.ef_search = 100
AS $fn$
BEGIN
  -- Dynamic SQL is planned with the actual parameter values on every call. A plain SQL function
  -- reuses a generic plan, which falls back to a sequential scan instead of the HNSW index.
  -- Filters are repeated in each branch (not a shared CTE) so vec uses HNSW and kw uses GIN.
  RETURN QUERY EXECUTE $q$
  WITH vec AS (
    SELECT f.chunk_id, row_number() OVER (ORDER BY f.embedding <=> $2) AS r
    FROM (
      SELECT c.chunk_id, c.embedding FROM content.chunks c
      WHERE $4 IN ('hybrid', 'vector')
        AND ($5 IS NULL OR c.brand = ANY($5))
        AND ($6 IS NULL OR c.tags && $6)
        AND ($7 IS NULL OR c.content_type = ANY($7))
        AND ($8 IS NULL OR c.published_at >= $8)
        AND ($9 IS NULL OR c.published_at <= $9)
      ORDER BY c.embedding <=> $2
      LIMIT $10
    ) f
  ),
  kw AS (
    SELECT f.chunk_id, row_number() OVER (ORDER BY f.rank DESC) AS r
    FROM (
      SELECT c.chunk_id, ts_rank_cd(c.tsv, q) AS rank
      FROM content.chunks c, websearch_to_tsquery('english', $1) q
      WHERE $4 IN ('hybrid', 'keyword') AND c.tsv @@ q
        AND ($5 IS NULL OR c.brand = ANY($5))
        AND ($6 IS NULL OR c.tags && $6)
        AND ($7 IS NULL OR c.content_type = ANY($7))
        AND ($8 IS NULL OR c.published_at >= $8)
        AND ($9 IS NULL OR c.published_at <= $9)
      ORDER BY rank DESC
      LIMIT $10
    ) f
  ),
  fused AS (
    SELECT chunk_id, vec.r AS vr, kw.r AS kr,
           coalesce(1.0 / ($11 + vec.r), 0) + coalesce(1.0 / ($11 + kw.r), 0) AS score
    FROM vec FULL OUTER JOIN kw USING (chunk_id)
  )
  SELECT c.chunk_id, c.article_id, c.chunk_index, c.brand, c.title, c.url, c.content_type,
         c.published_at, c.tags, c.chunk_text, fused.vr, fused.kr,
         (1 - (c.embedding <=> $2))::float8, fused.score::float8
  FROM fused JOIN content.chunks c USING (chunk_id)
  ORDER BY fused.score DESC
  LIMIT $3
  $q$
  USING q_text, q_emb, k, mode, brand_filter, tag_filter, type_filter, date_from, date_to, candidates, rrf_k;
END
$fn$;

-- Hybrid search over campaigns / media kit
CREATE OR REPLACE FUNCTION content.search_campaigns(
  q_text      TEXT,
  q_emb       VECTOR({DIM}),
  k           INT  DEFAULT 6,
  type_filter TEXT DEFAULT NULL,
  candidates  INT  DEFAULT 30,
  rrf_k       INT  DEFAULT 60
)
RETURNS TABLE (
  campaign_id TEXT, record_type TEXT, name TEXT, package_type TEXT, sponsor_category TEXT,
  brands TEXT[], target_audience TEXT, price_tier TEXT, price_range_usd TEXT, year INT,
  description TEXT, cosine_sim FLOAT8, score FLOAT8
)
LANGUAGE sql STABLE
AS $$
  WITH f AS (
    SELECT * FROM content.campaigns c WHERE type_filter IS NULL OR c.record_type = type_filter
  ),
  vec AS (
    SELECT f.campaign_id, row_number() OVER (ORDER BY f.embedding <=> q_emb) AS r
    FROM f ORDER BY f.embedding <=> q_emb LIMIT candidates
  ),
  kw AS (
    SELECT f.campaign_id, row_number() OVER (ORDER BY ts_rank_cd(f.tsv, q) DESC) AS r
    FROM f, websearch_to_tsquery('english', q_text) q
    WHERE f.tsv @@ q ORDER BY ts_rank_cd(f.tsv, q) DESC LIMIT candidates
  ),
  fused AS (
    SELECT campaign_id, coalesce(1.0 / (rrf_k + vec.r), 0) + coalesce(1.0 / (rrf_k + kw.r), 0) AS score
    FROM vec FULL OUTER JOIN kw USING (campaign_id)
  )
  SELECT c.campaign_id, c.record_type, c.name, c.package_type, c.sponsor_category, c.brands,
         c.target_audience, c.price_tier, c.price_range_usd, c.year, c.description,
         1 - (c.embedding <=> q_emb), fused.score
  FROM fused JOIN content.campaigns c USING (campaign_id)
  ORDER BY fused.score DESC
  LIMIT k
$$;
"""

with pg_connect(p, w) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    load_pgvector(conn)  # makes hnsw.* real settings before CREATE FUNCTION ... SET hnsw.iterative_scan
    conn.execute(DDL)
    print(conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone())
    for (t,) in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'content' ORDER BY 1"
    ):
        print("table:", t)
