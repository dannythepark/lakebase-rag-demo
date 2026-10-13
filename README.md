# Lakebase RAG demo on Databricks

An end-to-end Databricks demo showing that **one Lakebase Postgres database** (pgvector + Postgres full-text search) can replace a separate search service for retrieval-augmented generation. Everything is built in Python and deployed with one Asset Bundle: a Lakeflow Job that builds the data and the database, and a Databricks App that uses it.

Built for a customer conversation with a B2B trade-media publisher and published as a reusable template. The eight brands (FleetPulse, Upfit & Haul, Transit Forward, Yellow Route, Dealer Lot Report, Bay & Bench, Charge Depot, SafeMiles) are fictional, and all data is synthetic.

## What it shows

| Use case | Where |
|---|---|
| **Editor Search** (internal): "have we already published this?" Hybrid vector + keyword search across brands, tag facets, and an LLM coverage check with fresh angles | App → Editor Search · notebook `00` §6–8 |
| **Partner Assistant** (external): a grounded advertiser chatbot over a media kit and past campaigns, with citations, follow-up rewriting, code-computed price comparisons, chat history, session state and a **semantic answer cache**, all in the same database | App → Partner Assistant · notebook `00` §9 · `scripts/external_client.py` |
| **Market Lookup**: 4.5M company/fleet records served from Lakebase with covering indexes, in milliseconds, without a SQL warehouse | App → Market Lookup · notebook `06` |
| **Model flexibility**: two embedding models side by side, plus the AI Gateway config to bring an Azure OpenAI model | notebook `05` |
| **Cost transparency**: actual list-price spend from `system.billing.usage`, filtered to exactly these resources | notebook `07` · `docs/cost_onepager.md` |

The core path uses **GA features only**: Lakebase Autoscaling, pgvector, Foundation Model APIs (pay-per-token), AI Gateway, Databricks Apps, Lakeflow Jobs and Asset Bundles. See [`docs/feature_status.md`](docs/feature_status.md).

## Placeholders to fill in

| Placeholder | Where | What it is |
|---|---|---|
| `<workspace-host>` | `databricks.yml` | your workspace host, e.g. `my-workspace.cloud.databricks.com` |
| `<profile>` | commands below, `scripts/`, `docs/teardown.md` | your Databricks CLI profile |
| `demo_catalog` | `databricks.yml` (`catalog`), `src/rag_common.py` | a Unity Catalog catalog you can create schemas in (schema `rag_demo` is created for you) |

Models, sizes and names are bundle variables in `databricks.yml` (`embedding_endpoint`, `llm_endpoint`, `partner_llm_endpoint`, `n_articles`, `n_companies`, …). Check the models exist in your workspace region.

## The pipeline (one job, eight tasks)

```
generate_data (01) ─┐
                    ├─ embed_and_load (03) ─┬─ hybrid_search (04) ─┐
lakebase_setup (02) ┤                       └─ model_swap (05) ────┼─ search_showcase (00) ─ cost_model (07)
                    └─ market_intel (06) ──────────────────────────┘
```

| Task | Notebook | What it does |
|---|---|---|
| `generate_data` | `src/01_generate_data.py` | ~3,000 synthetic articles (news, blogs, webcast and videocast recaps) and ~150 media-kit / campaign records, written by an LLM with `ai_query`, into Delta |
| `lakebase_setup` | `src/02_lakebase_setup.py` | Creates the Lakebase project (autoscaling 0.5–4 CU, scale to zero after 5 min), enables pgvector, creates tables with a generated `tsvector` column, and the hybrid search SQL functions (vector + keyword, fused with Reciprocal Rank Fusion) |
| `embed_and_load` | `src/03_embed_and_load.py` | Chunk → embed (`ai_query`) → `COPY` into Postgres → build HNSW and GIN indexes after the load; logs timings and sizes to `run_metrics` |
| `market_intel` | `src/06_market_intel.py` | 4.5M synthetic companies in Delta → Lakebase (synced table, or a `COPY` fallback if you lack `CREATE CATALOG`) → covering indexes → Lakebase vs. Delta benchmark |
| `hybrid_search` | `src/04_hybrid_search.py` | Vector vs. keyword vs. hybrid on six queries, filters, query plans, latency benchmark |
| `model_swap` | `src/05_model_swap.py` | Re-embeds a subset with a second model and compares results; AI Gateway config for Azure OpenAI |
| `search_showcase` | `src/00_lakebase_search_showcase.py` | **Start here.** A guided, read-only walkthrough of Lakebase search, one concept per cell |
| `cost_model` | `src/07_cost_model.py` | List-price spend from the billing system tables for exactly these resources |

## Layout
```
databricks.yml                bundle: variables + target
resources/                    the pipeline job, a standalone market-data job, and the app
src/rag_common.py             shared config and Lakebase / embedding helpers
src/00_…07_*.py               the notebooks above
app/backend/                  FastAPI: search, partner chat, cache, market lookup (psycopg pool with OAuth token refresh)
app/frontend/                 React + Vite UI (build output goes to app/static)
scripts/external_client.py    call the Partner Assistant API from outside Databricks with OAuth
scripts/grant_app_access.py   grant the app's service principal read access to the content and market schemas
docs/                         cost one-pager, feature status, teardown
```

## Deploy from scratch
```bash
P=<profile>
(cd app/frontend && npm install && npx vite build)      # builds app/static
databricks bundle deploy --profile $P
databricks bundle run content_pipeline --profile $P       # the full 8-task DAG; ~1–2 h, mostly LLM article generation and embeddings
databricks bundle run content_app --profile $P            # deploy and start the app
python scripts/grant_app_access.py --profile $P           # once the app's service principal exists
```

Deploy the app before running it locally, so the app's service principal creates and owns the `app` schema.

## Run the app locally
```bash
cd app && DATABRICKS_CONFIG_PROFILE=<profile> uvicorn backend.main:app --port 8000
# frontend hot reload: cd app/frontend && npx vite   (proxies /api to :8000)
```

## Call it from outside Databricks
```bash
python scripts/external_client.py --profile <profile> \
  --app-url https://content-assistant-<workspace-id>.<cloud>.databricksapps.com \
  "What results have telematics vendors seen from past campaigns?"
```
In production, use a service principal (OAuth M2M) with `CAN_USE` on the app.

## Notes from running it
- Warm hybrid search: ~3–5 ms inside Postgres, ~7–9 ms from a notebook, ~9–18 ms from the app. The first queries after scale-to-zero take 0.5–4 s while compute wakes.
- Market lookups: ~3 ms p50 / 8 ms p95 in Lakebase vs. ~850 ms for the same query on Delta/Spark.
- Pay-per-token batch embedding throughput varies with shared capacity (38 s on one run, ~60 min on another for the same 2.3M tokens). Use provisioned throughput for large backfills.
- The always-on app is the largest cost line at demo scale; Lakebase itself costs cents with scale-to-zero. Stop the app when you're not using it.

## Teardown
See [`docs/teardown.md`](docs/teardown.md). Deleting the Lakebase project is permanent.
