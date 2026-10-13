# Cost model: Lakebase RAG vs. a separate search stack (list price, estimate)

> **Status:** usage *units* are measured or derived from the demo. Dollar figures come from `src/07_cost_model.py`,
> which reads `system.billing.usage` × `list_prices` once billing lands (a few hours' delay). Run it in your own
> workspace to see your cloud's prices. All numbers are list price, before any discounts.

## Measured demo spend (Oct 1–12, 2026, AWS us-east-1 list price, from `system.billing.usage` × `list_prices`)

| Component | Usage | List $ | Notes |
|---|---|---|---|
| Databricks App (Medium) | 46.7 DBU over 95 billed hours | **$44.38** | ~$0.47/hour while running. **The biggest line by far**: stop it when not in use |
| Serverless jobs (data generation, embedding, load) | 10.5 DBU | $4.72 | Mostly one-time demo build |
| Lakebase storage | 8.8 DSU-units | $0.20 | ~1.7 GB incl. the 4.5M-row market table |
| Lakebase compute | 0.23 DBU in 5 billed hours | $0.12 | Scale-to-zero at work. ⚠️ Looks low for the load work done; recheck with `07_cost_model` before quoting |
| Model serving (FMAPI) attributed to the demo | — | ~$0.30 | ⚠️ Attribution is partial (`ai_query` generation tokens aren't all tagged to the job); don't quote |
| **Total (tagged)** | | **≈ $49.50** | |

**Read this as:** at demo scale the database is cents per day, and the always-on app compute dominates. For production, size the app by traffic and keep Lakebase on scale-to-zero for internal-only tools.

## What drives cost

| Component | Unit | Demo (measured) | Production (assumption) | Notes |
|---|---|---|---|---|
| Lakebase compute | CU-hours | 0.5–4 CU autoscaling, suspends after 5 idle min | ~0.5–1 CU most hours; ~2 CU at peak. Business hours only with scale-to-zero, or 24×7 for the external bot | Hybrid search: 3.2 ms p50 / 4.8 ms p95 server-side; 6.7 / 8.9 ms from a notebook; 9 / 18 ms from the app when warm (130–500 ms on cold first queries); the database is not the bottleneck |
| Lakebase storage | GB-month | **0.11 GB** content (3K articles, 6K chunks; HNSW index 47 MB) + ~1.5 GB market data (4.5M rows + covering indexes) | ~100K articles (assumed archive) → ~200K chunks ≈ **~6 GB** content + market data | Measured with `pg_total_relation_size`, logged to `run_metrics` |
| Embeddings, backfill | tokens (one-time) | **~2.3M tokens**: 38 s on Oct 5, ~60 min on Oct 13 (3K articles → 6,002 chunks) | ~100K articles × ~775 tokens ≈ **~78M tokens one-time** | `databricks-gte-large-en`, pay-per-token; plus `COPY` 8 s and index build 3.5 s. Pay-per-token batch throughput varies with shared capacity; size dedicated throughput for a large one-time backfill |
| Embeddings, ongoing | tokens / month | n/a | new articles (~2K/mo × 775) + queries (25K × ~15) ≈ **~2M / month** | |
| LLM: Partner Assistant | tokens / question | ~3.5K in / ~350 out on `databricks-claude-sonnet-4-6`, plus ~1K in / ~60 out for the follow-up rewrite on Haiku | 5K questions/mo × (1 − cache hit rate) | A cache hit skips retrieval and both LLM calls: <0.5 s vs. 5–7 s measured |
| LLM: Editor coverage check | tokens / search | ~1.2K in / ~250 out | 20K searches/mo (can be made on-demand) | `databricks-claude-haiku-4-5` |
| Databricks App | app-hours | Medium, running continuously | Medium 24×7 = 730 h/mo | Fixed cost; stop it when not needed |
| Market lookups | Lakebase CU (shared) | 4.5M rows, **3.0 ms p50 / 8.0 ms p95** per lookup vs. **846 ms** for the same query on Delta/Spark; 13–60 ms through the deployed app | same | **Replaces a large SQL warehouse** for repeated interactive lookups |
| One-time demo data generation | `ai_query` LLM tokens | ~3K articles × ~900 out tokens | **$0**: a real publisher's content already exists | Demo-only line item |

## Monthly estimate (fill in)

| Line | Units/mo | List $/unit (your cloud and region) | $/mo |
|---|---|---|---|
| Lakebase compute | ___ CU-h | ___ | ___ |
| Lakebase storage | ___ GB | ___ | ___ |
| Embeddings | ~2.2M tokens | ___ | ___ |
| LLM input | ___ tokens | ___ | ___ |
| LLM output | ___ tokens | ___ | ___ |
| Databricks App (Medium) | 730 h | ___ | ___ |
| **Total** | | | ___ |

## A typical separate-search stack, for comparison (fill in from your own bill)

| Line | Today | Replaced by |
|---|---|---|
| Azure AI Search (tier ___, ___ replicas/partitions) | $___ /mo | Lakebase pgvector + full-text |
| Azure OpenAI embeddings | $___ /mo | FMAPI embeddings (or keep Azure OpenAI through AI Gateway) |
| Azure OpenAI chat | $___ /mo | FMAPI LLM |
| Orchestration functions | $___ /mo | Python in the Databricks App |
| Separate Postgres (AI product logs) | $___ /mo | Lakebase (same database) |
| SQL warehouse for repeated 4.5M-row lookups | $___ /mo | Lakebase synced table |

## Cost levers to point out
- **Scale to zero** on Lakebase (5 min idle) and stop the app off-hours for internal-only tools.
- **Semantic answer cache:** every cache hit skips retrieval and both LLM calls.
- **Model choice is a parameter.** A smaller or cheaper LLM can serve the partner bot, and an existing Azure OpenAI deployment can be used through AI Gateway.
- **Coverage check on demand** (a button instead of every search) cuts LLM calls for Editor Search by ~90%.
