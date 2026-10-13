# Features used in the demo, by release status

The **core path (Editor Search + Partner Assistant)** uses only generally available features, so it runs on a pay-as-you-go account with no preview enrollment. Release status changes, so check the docs release notes for your cloud and region before relying on this table.

| Feature | Where it's used | Status (Oct 2026) | Notes |
|---|---|---|---|
| Lakebase Postgres (Autoscaling) | All app data: vectors, full-text, sessions, cache | GA on AWS and Azure | Autoscaling 0.5–4 CU, scale to zero, branching |
| pgvector extension (0.8.0) | `VECTOR(1024)`, HNSW index, iterative filtered scans | Postgres extension on Lakebase | Not a separate product: `CREATE EXTENSION vector` |
| Postgres full-text search (`tsvector`, GIN) | Keyword half of hybrid search | Core Postgres | |
| Foundation Model APIs, pay-per-token | `databricks-gte-large-en`, `databricks-qwen3-embedding-0-6b`, `databricks-claude-haiku-4-5`, `databricks-claude-sonnet-4-6` | GA | Availability varies by region; some regions need cross-geo routing |
| `ai_query` | Batch generation and embedding in notebooks | GA | |
| AI Gateway external models (e.g. Azure OpenAI) | Optional model swap in `05_model_swap` | GA | Shown as config only |
| Databricks Apps | FastAPI + React app | GA | |
| Lakeflow Jobs (serverless) | Content pipeline job | GA | |
| Databricks Asset Bundles | Deploy and teardown | GA | |
| Lakebase Search (`lakebase_vector` / `lakebase_ann`, `lakebase_text` BM25) | Not used; an upgrade path for the vector and keyword indexes | GA | Enabling it restarts project computes and can't be undone |
| Lakebase synced tables | Market Lookup (4.5M rows) | Check for your region | Optional; needs `CREATE CATALOG` to register the database in Unity Catalog. The notebook falls back to `COPY` |

**Not used, intentionally:** Mosaic AI Vector Search and Agent Bricks Knowledge Assistant. Both are good managed alternatives (Vector Search for very large or fully managed Delta-synced indexes, Knowledge Assistant for zero-code RAG). The point here is that one Lakebase database can replace a separate search tier.
