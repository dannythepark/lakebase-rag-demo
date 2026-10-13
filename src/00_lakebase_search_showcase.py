# Databricks notebook source
# MAGIC %md
# MAGIC # Lakebase search, step by step
# MAGIC
# MAGIC **For:** architects and product owners evaluating Lakebase for search · **Runs on:** serverless notebook, about 5 minutes top to bottom · **Changes nothing:** read-only against the demo data
# MAGIC
# MAGIC A typical publisher's content search today is spread across **Azure AI Search** (the index), **Azure OpenAI** (embeddings and chat) and **custom orchestration functions**
# MAGIC (the glue). This notebook shows the same capability running inside **one Postgres database, Lakebase**, using plain SQL and Python.
# MAGIC Every cell explains what it does and why it matters, so you can rerun it yourself.
# MAGIC
# MAGIC | # | Section | Answers the question |
# MAGIC |---|---|---|
# MAGIC | 1 | The database | *"What is Lakebase, and does an idle project cost money?"* |
# MAGIC | 2 | Anatomy of a searchable row | *"Where do the vectors live?"* |
# MAGIC | 3 | Question → vector | *"Which embedding model, and can I choose it?"* |
# MAGIC | 4 | Semantic search (pgvector) | *"Can it find meaning, not just words?"* |
# MAGIC | 5 | Keyword search (Postgres full-text) | *"What about exact jargon and product names?"* |
# MAGIC | 6 | Hybrid search in one SQL statement | *"How do I get both, without a separate search service?"* |
# MAGIC | 7 | Filters and tag facets | *"Which brands and tags were used for this topic?"* |
# MAGIC | 8 | Use case 1: "Have we already published this?" | Editors |
# MAGIC | 9 | Use case 2: Partner assistant + semantic answer cache | Advertisers (external) |
# MAGIC | 10 | Under the hood: indexes and query plans | *"Is it really using an index? How fast is it?"* |
# MAGIC | 11 | Use case 3: 4.5M company/fleet records | *"Fast lookups without a large SQL warehouse"* |
# MAGIC | 12 | Swap the embedding model | *"Am I locked into a model?"* |
# MAGIC
# MAGIC > All data is synthetic: 8 fictional fleet and transportation brands, about 3,000 articles, 150 media-kit and campaign records, and 4.5M companies.
# MAGIC > The pipeline that built it (`01`–`06` notebooks) is one Lakeflow Job in the same bundle.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup: connect with a standard Postgres driver
# MAGIC Lakebase is Postgres, so we connect with `psycopg`, the same driver any Python app uses. There is no password:
# MAGIC the notebook asks Databricks for a short-lived **OAuth token** for the signed-in user, and Unity Catalog identity carries through to the database.

# COMMAND ----------

import re
import time

import pandas as pd
from databricks.sdk import WorkspaceClient

from rag_common import Lakebase, Timer, embed, endpoint_path, get_params, pg_connect, vec_literal

p = get_params(dbutils)
w = WorkspaceClient()
EMB = p["embedding_endpoint"]  # databricks-gte-large-en by default; a parameter, not code
LLM = p["llm_endpoint"]

conn = pg_connect(p, w)
conn.execute("SET hnsw.ef_search = 100")  # HNSW search breadth: higher means better recall, slightly slower

pd.set_option("display.max_colwidth", 90)


def show(df):
    try:
        display(df)  # noqa: F821 (Databricks notebook)
    except NameError:
        print(df.to_string(index=False))


def sql(query, params=None):
    """Run SQL against Lakebase and return a DataFrame."""
    cur = conn.execute(query, params)
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def ask_llm(prompt, system="Be concise.", max_tokens=300, endpoint=LLM):
    r = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.2})
    return r["choices"][0]["message"]["content"]


def plan(query, params=None):
    """EXPLAIN ANALYZE, with long vector literals shortened so the plan is readable."""
    rows = conn.execute("EXPLAIN (ANALYZE, BUFFERS OFF) " + query, params).fetchall()
    return "\n".join(re.sub(r"'\[[^\]]{40,}\]'", "'[...1024 numbers...]'", r[0]) for r in rows)


print("connected to", conn.info.host, "| database", conn.info.dbname, "| user", conn.info.user)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · The database
# MAGIC A Lakebase **project** is a managed Postgres 17 database. Compute **autoscales** between a minimum and maximum number of
# MAGIC capacity units (CU), and **scales to zero** after a few idle minutes. When it's suspended you pay for storage only, and the
# MAGIC next query wakes it automatically.
# MAGIC
# MAGIC That answers a common question: *an idle project doesn't keep burning compute if scale-to-zero is on.*

# COMMAND ----------

ep = Lakebase(w).get(endpoint_path(p))["status"]
show(pd.DataFrame([{
    "project": p["lakebase_project"],
    "autoscaling (CU)": f'{ep["autoscaling_limit_min_cu"]} – {ep["autoscaling_limit_max_cu"]}',
    "scale to zero after idle": ep.get("suspend_timeout_duration"),
    "state right now": ep.get("current_state"),
    "postgres": conn.execute("SHOW server_version").fetchone()[0].split()[0],
    "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0],
}]))

# COMMAND ----------

# MAGIC %md
# MAGIC **Everything in one place.** Search content, the media kit, the partner chat history and answer cache, and the market data
# MAGIC are all tables in the same database. There's no separate search service to size, pay for or keep in sync.

# COMMAND ----------

show(sql("""
SELECT n.nspname AS schema, c.relname AS table_name,
       CASE WHEN c.reltuples < 0 THEN NULL ELSE c.reltuples::bigint END AS approx_rows,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS size_incl_indexes,
       CASE n.nspname WHEN 'content' THEN 'search content (vectors + full text)'
                      WHEN 'app' THEN 'partner chat: sessions, history, answer cache'
                      WHEN 'market' THEN '4.5M company/fleet records' END AS purpose
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND n.nspname IN ('content', 'app', 'market')
ORDER BY n.nspname, pg_total_relation_size(c.oid) DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Anatomy of a searchable row
# MAGIC Each article is split into chunks of about 350 words. One row in `content.chunks` holds **everything search needs**:
# MAGIC - the text and its metadata (brand, date, content type, tags, URL)
# MAGIC - `embedding`: a `VECTOR(1024)` column from pgvector, which captures the *meaning* of the chunk
# MAGIC - `tsv`: a `TSVECTOR` column that Postgres **generates automatically** from the title and text, for keyword search
# MAGIC
# MAGIC Because `tsv` is a generated column, the keyword index can't drift out of date. Insert or update a row, and it's searchable both ways.

# COMMAND ----------

show(sql("""
SELECT column_name, data_type, udt_name,
       CASE column_name WHEN 'embedding' THEN 'pgvector: meaning of the chunk (1024 numbers)'
                        WHEN 'tsv' THEN 'GENERATED ALWAYS from title + text: keyword index'
                        WHEN 'tags' THEN 'Postgres array: filter with && (overlaps)' END AS note
FROM information_schema.columns
WHERE table_schema = 'content' AND table_name = 'chunks' ORDER BY ordinal_position
"""))

# COMMAND ----------

show(sql("""
SELECT brand, published_at, content_type, tags, title,
       left(chunk_text, 160) || '…' AS chunk_text,
       left(embedding::text, 60) || '…' AS embedding,
       left(tsv::text, 80) || '…' AS tsv
FROM content.chunks WHERE brand = 'Charge Depot' ORDER BY published_at DESC LIMIT 2
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Turning a question into a vector
# MAGIC At query time we embed the question with **the same model** used for the content. The model is a Databricks-hosted
# MAGIC Foundation Model API endpoint, called like any REST API. It's a **parameter** (`embedding_endpoint`): set it to another
# MAGIC Databricks model, or to your existing Azure OpenAI deployment registered in AI Gateway (section 12).

# COMMAND ----------

question = "EV charging infrastructure for depot fleets"
with Timer() as t:
    q_vec = embed(w, EMB, [question])[0]
print(f"model: {EMB}\ndimensions: {len(q_vec)}\nfirst 6 values: {[round(x, 3) for x in q_vec[:6]]}\nembedding call: {t.ms:.0f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Semantic search with pgvector
# MAGIC `<=>` is pgvector's **cosine distance** operator. "Find the closest chunks to my question" is plain SQL:
# MAGIC `ORDER BY embedding <=> question LIMIT k`.
# MAGIC
# MAGIC The query below is a *paraphrase*: no article uses the words "utility money". Semantic search still finds articles about
# MAGIC **make-ready programs and utility incentives**, because it matches meaning, not words.

# COMMAND ----------

paraphrase = "how do I get utility money to pay for chargers"
p_vec = vec_literal(embed(w, EMB, [paraphrase])[0])

show(sql("""
SELECT round((1 - (embedding <=> %(v)s::vector))::numeric, 3) AS similarity, brand, published_at, title
FROM content.chunks
ORDER BY embedding <=> %(v)s::vector
LIMIT 10
""", {"v": p_vec}).drop_duplicates("title").head(5))  # several chunks of one article can match; show each article once

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Keyword search with Postgres full-text search
# MAGIC Semantic search can blur exact terms: product names, acronyms, legal jargon. Postgres has built-in full-text search for that:
# MAGIC - `websearch_to_tsquery` accepts what a user would type into a search box (quotes, `or`, `-exclude`)
# MAGIC - `ts_rank_cd` scores matches; `ts_headline` highlights **why** a document matched
# MAGIC
# MAGIC It's served by a **GIN index** on the `tsv` column, the same kind of inverted index a search engine uses.

# COMMAND ----------

show(sql("""
SELECT round(ts_rank_cd(tsv, q)::numeric, 3) AS rank, brand, published_at, title,
       ts_headline('english', chunk_text, q, 'MaxFragments=1, MaxWords=18, MinWords=8, StartSel=«, StopSel=»') AS why_it_matched
FROM content.chunks, websearch_to_tsquery('english', %(q)s) q
WHERE tsv @@ q
ORDER BY rank DESC
LIMIT 5
""", {"q": '"nuclear verdicts" insurance'}))

# COMMAND ----------

# MAGIC %md
# MAGIC And the flip side: on the paraphrase from section 4, keyword search finds **nothing**, because no document contains those words together.
# MAGIC This is why you want both.

# COMMAND ----------

print("keyword hits for the paraphrase:",
      conn.execute("SELECT count(*) FROM content.chunks WHERE tsv @@ websearch_to_tsquery('english', %s)", (paraphrase,)).fetchone()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Hybrid search: both, fused, in one SQL statement
# MAGIC **Reciprocal Rank Fusion (RRF)** is the standard way to combine ranked lists, and it's what Azure AI Search uses for its own hybrid mode.
# MAGIC Each result scores `1 / (60 + rank)` in each list it appears in, and the scores are added. Results that rank well in *both*
# MAGIC lists rise to the top, and you don't need to tune weights.
# MAGIC
# MAGIC Here is the **entire** hybrid search. Read it top to bottom:
# MAGIC 1. `vec`: top 60 chunks by meaning (HNSW index)
# MAGIC 2. `kw`: top 60 chunks by keywords (GIN index)
# MAGIC 3. `fused`: RRF over the two lists
# MAGIC
# MAGIC In the app, this same statement is wrapped as a SQL function, `content.search_chunks(...)`, so the app sends one call.

# COMMAND ----------

HYBRID_SQL = """
WITH vec AS (                                     -- 1. semantic: nearest neighbours
  SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(v)s::vector) AS r
  FROM (SELECT chunk_id, embedding FROM content.chunks
        ORDER BY embedding <=> %(v)s::vector LIMIT 60) t
),
kw AS (                                           -- 2. keyword: full-text matches
  SELECT chunk_id, row_number() OVER (ORDER BY rank DESC) AS r
  FROM (SELECT chunk_id, ts_rank_cd(tsv, q) AS rank
        FROM content.chunks, websearch_to_tsquery('english', %(q)s) q
        WHERE tsv @@ q ORDER BY rank DESC LIMIT 60) t
),
fused AS (                                        -- 3. reciprocal rank fusion
  SELECT chunk_id, vec.r AS vector_rank, kw.r AS keyword_rank,
         coalesce(1.0 / (60 + vec.r), 0) + coalesce(1.0 / (60 + kw.r), 0) AS rrf_score
  FROM vec FULL OUTER JOIN kw USING (chunk_id)
)
SELECT round(rrf_score, 4) AS rrf_score, vector_rank, keyword_rank, c.brand, c.published_at, c.title
FROM fused JOIN content.chunks c USING (chunk_id)
ORDER BY rrf_score DESC
LIMIT %(k)s
"""

show(sql(HYBRID_SQL, {"q": question, "v": vec_literal(q_vec), "k": 16}).drop_duplicates("title").head(8))  # one row per article

# COMMAND ----------

# MAGIC %md
# MAGIC **Side by side.** The top 3 article titles for each mode, across three kinds of query. The `vector_rank` and `keyword_rank`
# MAGIC columns above show where each result came from. The table below shows the behaviour: keyword search wins on jargon, vector search wins
# MAGIC on paraphrase, and hybrid holds up on both.

# COMMAND ----------

compare = []
for q in [question, '"nuclear verdicts" insurance', paraphrase]:
    v = vec_literal(embed(w, EMB, [q])[0])
    for mode in ["vector", "keyword", "hybrid"]:
        rows = conn.execute("SELECT title FROM content.search_chunks(%s, %s::vector, k => 20, mode => %s)", (q, v, mode)).fetchall()
        titles = list(dict.fromkeys(r[0] for r in rows))[:3]  # chunks → distinct articles
        compare.append({"query": q, "mode": mode, "hits": len(rows),
                        **{f"#{i + 1}": titles[i] if i < len(titles) else "—" for i in range(3)}})
show(pd.DataFrame(compare))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Filters and tag facets are just SQL
# MAGIC Brand, date and tag filters are ordinary `WHERE` clauses, applied *inside* both halves of the hybrid search. pgvector 0.8's
# MAGIC **iterative index scan** keeps searching the HNSW index until it finds enough rows that pass the filter, so filtered
# MAGIC searches don't come back half-empty.

# COMMAND ----------

show(sql("""
SELECT brand, published_at, content_type, tags, title
FROM content.search_chunks(%(q)s, %(v)s::vector, k => 6,
                           brand_filter => ARRAY['Transit Forward'], date_from => DATE '2024-01-01')
""", {"q": question, "v": vec_literal(q_vec)}).drop_duplicates("title"))

# COMMAND ----------

# MAGIC %md
# MAGIC Editors want to know **which tags were used** on related content. That's an aggregate over the search results,
# MAGIC computed in the same query.

# COMMAND ----------

show(sql("""
WITH hits AS (SELECT DISTINCT article_id, tags FROM content.search_chunks(%(q)s, %(v)s::vector, k => 40))
SELECT tag, count(*) AS articles
FROM hits, unnest(tags) AS tag
GROUP BY tag ORDER BY articles DESC LIMIT 10
""", {"q": question, "v": vec_literal(q_vec)}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Use case 1: "Have we already published this?" (editors)
# MAGIC An editor pitches a story. We run hybrid search, collapse chunks to articles, and summarise coverage **by brand and year**.
# MAGIC Then a small, inexpensive LLM (`databricks-claude-haiku-4-5`) reads only those results and writes a short verdict with fresh angles.
# MAGIC This is what the **Editor Search** tab in the app does.

# COMMAND ----------

pitch = "EV charging infrastructure for depot fleets"
pitch_vec = vec_literal(embed(w, EMB, [pitch])[0])
related = sql("""
SELECT DISTINCT ON (article_id) article_id, brand, published_at, content_type, title, left(chunk_text, 300) AS snippet, score
FROM content.search_chunks(%(q)s, %(v)s::vector, k => 40)
ORDER BY article_id, score DESC
""", {"q": pitch, "v": pitch_vec}).sort_values("score", ascending=False).head(12)

coverage = related.assign(year=pd.to_datetime(related.published_at).dt.year) \
                  .pivot_table(index="brand", columns="year", values="article_id", aggfunc="count", fill_value=0)
show(coverage.reset_index())

# COMMAND ----------

listing = "\n".join(f"- [{r.brand}, {r.published_at}, {r.content_type}] {r.title} :: {r.snippet}" for r in related.head(8).itertuples())
print(ask_llm(
    f'An editor is pitching: "{pitch}"\n\nMost relevant previously published pieces:\n{listing}\n\n'
    "In at most 90 words: (1) say whether this is already well covered, how often, by which brands and how recently; "
    '(2) suggest 2 fresh angles not yet covered. One short paragraph, then "Fresh angles:" and two bullets.',
    system="You help B2B trade-media editors avoid duplicate coverage. Only use the search results provided."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Use case 2: Partner assistant (external) and the semantic answer cache
# MAGIC **Retrieval-augmented generation (RAG)** in three steps, with nothing hidden:
# MAGIC 1. **Retrieve** from Lakebase: media-kit offerings and past campaign case studies (`content.search_campaigns`), plus editorial coverage that proves the audience
# MAGIC 2. **Ground**: number every source and tell the model to answer *only* from them, with citations
# MAGIC 3. **Generate** with a stronger model for advertiser-facing answers (`databricks-claude-sonnet-4-6`)
# MAGIC
# MAGIC The app adds chat history, follow-up rewriting, and computed entry prices so the model never compares numbers itself. The core is below.

# COMMAND ----------

ask = "We sell AI dash cams to mid-size fleets. What sponsorship options reach fleet safety managers?"
ask_vec = vec_literal(embed(w, EMB, [ask])[0])

with Timer() as t_ret:
    sources = sql("SELECT record_type, name, brands, price_range_usd, target_audience, description "
                  "FROM content.search_campaigns(%s, %s::vector, k => 5)", (ask, ask_vec))
print(f"retrieved {len(sources)} media-kit/campaign records from Lakebase in {t_ret.ms:.0f} ms (round trip)")
show(sources[["record_type", "name", "brands", "price_range_usd"]])

# COMMAND ----------

context = "\n\n".join(f"[{i}] {r.record_type}: {r.name} | brands: {', '.join(r.brands)} | price: ${r.price_range_usd} | "
                      f"audience: {r.target_audience}\n{r.description}" for i, r in enumerate(sources.itertuples(), 1))
with Timer() as t_llm:
    answer = ask_llm(f"Context:\n\n{context}\n\nQuestion: {ask}", max_tokens=400, endpoint=p["partner_llm_endpoint"],
                     system="You are the partnerships assistant for a B2B trade-media company. Answer ONLY from the numbered context, "
                            "cite sources like [1], never invent prices or results, and keep it under 150 words.")
print(answer, f"\n\n(LLM {t_llm.ms / 1000:.1f} s)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Semantic answer cache.** External bots get the same questions all day, worded slightly differently. Before running RAG, the app
# MAGIC embeds the question and looks for a near-identical question it has **already answered**, stored in `app.answer_cache` in the same
# MAGIC database. If cosine similarity is ≥ 0.95, it returns the stored answer: no retrieval and no LLM call, in well under a second.
# MAGIC
# MAGIC Below, Postgres computes the similarity between two wordings directly:

# COMMAND ----------

reworded = "We sell AI dashcams to mid-sized fleets. Which sponsorship options reach fleet safety managers?"
unrelated = "What does a newsletter takeover cost on the school bus brand?"
show(sql("""SELECT %(b)s AS compared_with, round((1 - (%(a)s::vector <=> %(bv)s::vector))::numeric, 4) AS similarity,
                   (1 - (%(a)s::vector <=> %(bv)s::vector)) >= 0.95 AS cache_hit
            UNION ALL
            SELECT %(c)s, round((1 - (%(a)s::vector <=> %(cv)s::vector))::numeric, 4),
                   (1 - (%(a)s::vector <=> %(cv)s::vector)) >= 0.95""",
         {"a": ask_vec, "b": reworded, "bv": vec_literal(embed(w, EMB, [reworded])[0]),
          "c": unrelated, "cv": vec_literal(embed(w, EMB, [unrelated])[0])}))

# COMMAND ----------

# MAGIC %md
# MAGIC The app's cache lookup is one indexed query, and the conversation log sits next to it. This is the same database that could also
# MAGIC hold AI-product interaction logs that often live in a separate Postgres today.
# MAGIC ```sql
# MAGIC SELECT answer, citations, 1 - (question_embedding <=> $1) AS similarity
# MAGIC FROM app.answer_cache ORDER BY question_embedding <=> $1 LIMIT 1;
# MAGIC ```

# COMMAND ----------

try:
    show(sql("""SELECT (SELECT count(*) FROM app.sessions) AS sessions, (SELECT count(*) FROM app.messages) AS messages,
                       (SELECT count(*) FROM app.answer_cache) AS cached_answers,
                       (SELECT coalesce(sum(hit_count), 0) FROM app.answer_cache) AS cache_hits"""))
except Exception as e:  # the app's service principal owns the `app` schema; your user may not have been granted read access
    print("app schema not readable by this user:", str(e).splitlines()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Under the hood: indexes and query plans
# MAGIC `EXPLAIN ANALYZE` shows how Postgres actually ran a query and the **server-side execution time**, with network excluded.
# MAGIC
# MAGIC **Keyword search** uses the GIN index (`Bitmap Index Scan on chunks_tsv_gin`):

# COMMAND ----------

print(plan("SELECT chunk_id FROM content.chunks WHERE tsv @@ websearch_to_tsquery('english', %s)", ('"nuclear verdicts"',)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Vector search** uses the HNSW index (`Index Scan using chunks_embedding_hnsw`): it walks a graph of neighbouring vectors
# MAGIC instead of comparing the question with every row, in well under a millisecond.
# MAGIC
# MAGIC *A note on table size:* Postgres picks whatever plan it estimates is cheapest. With only ~6,000 chunks, it sometimes decides
# MAGIC an exact scan of every row is cheap enough (about 20 ms, and exact). If you see `Seq Scan` below, that's why. With a real archive
# MAGIC of hundreds of thousands of chunks, it uses the index. The second plan disables the sequential scan for one statement, so you always see the index.

# COMMAND ----------

knn = "SELECT chunk_id FROM content.chunks ORDER BY embedding <=> %s::vector LIMIT 10"
chosen = plan(knn, (vec_literal(q_vec),))
print("— planner's choice —\n" + chosen)
if "hnsw" not in chosen:
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        print("\n— same query, HNSW index —\n" + plan(knn, (vec_literal(q_vec),)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Speed of the full hybrid search** (both indexes, fusion, join back to the text). We reuse precomputed question vectors so only
# MAGIC the database is timed. Server time comes from Postgres itself; round trip adds the network between this notebook and the database.

# COMMAND ----------

bench_qs = [question, '"nuclear verdicts" insurance', paraphrase, "school bus driver shortage", "telematics ROI for small fleets"]
bench_vecs = {q: vec_literal(embed(w, EMB, [q])[0]) for q in bench_qs}
server_ms, roundtrip_ms = [], []
for _ in range(10):
    for q, v in bench_vecs.items():
        out = plan("SELECT * FROM content.search_chunks(%s, %s::vector, k => 10)", (q, v))
        server_ms.append(float(re.search(r"Execution Time: ([\d.]+)", out).group(1)))
        with Timer() as t:
            conn.execute("SELECT * FROM content.search_chunks(%s, %s::vector, k => 10)", (q, v)).fetchall()
        roundtrip_ms.append(t.ms)
pctl = lambda xs, k: round(sorted(xs)[min(len(xs) - 1, int(len(xs) * k))], 1)
show(pd.DataFrame([
    {"measure": "hybrid search, server time (Postgres)", "p50 ms": pctl(server_ms, .5), "p95 ms": pctl(server_ms, .95)},
    {"measure": "hybrid search, notebook round trip", "p50 ms": pctl(roundtrip_ms, .5), "p95 ms": pctl(roundtrip_ms, .95)},
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 · Use case 3: 4.5M company and fleet records
# MAGIC *"How many companies in Texas run 50–200 vehicles?"* Run on a large SQL warehouse, this kind of repeated lookup gets expensive fast.
# MAGIC
# MAGIC Here, the Delta table in Unity Catalog stays the source of truth, and a copy is served from Lakebase with **covering indexes**.
# MAGIC In production that copy is a **synced table**, a managed Delta → Postgres pipeline. The lookup is an index-only scan:
# MAGIC Postgres answers from the index without touching the table, on the same small compute as everything above.
# MAGIC
# MAGIC Heavy, ad-hoc analytics still belong on a SQL warehouse or Genie. Repeated interactive lookups belong here.

# COMMAND ----------

market_q = """SELECT count(*) AS companies, sum(fleet_size) AS vehicles, round(avg(fleet_size), 1) AS avg_fleet
              FROM market.companies WHERE state = %s AND fleet_size BETWEEN %s AND %s"""
show(sql(market_q, ("TX", 50, 200)))
print(plan(market_q, ("TX", 50, 200)))

# COMMAND ----------

show(sql("""SELECT state, count(*) AS companies_50_200_vehicles, round(100.0 * avg(has_telematics::int), 1) AS pct_with_telematics
            FROM market.companies WHERE state = ANY(%s) AND fleet_size BETWEEN 50 AND 200
            GROUP BY state ORDER BY companies_50_200_vehicles DESC""", (["TX", "CA", "FL", "OH", "GA"],)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12 · Swap the embedding model
# MAGIC Notebook `05_model_swap` re-embedded 500 articles with a **second** model (`databricks-qwen3-embedding-0-6b`) into a parallel table.
# MAGIC Same SQL, same index type, different model. (Both happen to produce 1024 numbers; the column size `VECTOR(n)` is a parameter
# MAGIC for models that don't.) To keep the **Azure OpenAI** embeddings you already pay for, register them
# MAGIC once as an AI Gateway external model and point `embedding_endpoint` at it. Nothing else changes. (The config is in `05_model_swap`.)

# COMMAND ----------

ALT = p["alt_embedding_endpoint"]
q = "telematics ROI for small fleets"
a = sql("""SELECT c.title FROM content.chunks c JOIN content.chunks_alt x USING (chunk_id)
           ORDER BY c.embedding <=> %s::vector LIMIT 15""", (vec_literal(embed(w, EMB, [q])[0]),)).drop_duplicates("title").head(5)
b = sql("""SELECT c.title FROM content.chunks_alt x JOIN content.chunks c USING (chunk_id)
           ORDER BY x.embedding <=> %s::vector LIMIT 15""", (vec_literal(embed(w, ALT, [q])[0]),)).drop_duplicates("title").head(5)
alt_dim = conn.execute("SELECT vector_dims(embedding) FROM content.chunks_alt LIMIT 1").fetchone()[0]
show(pd.DataFrame({"rank": range(1, 6), f"{EMB} (1024 dims)": a.title.values, f"{ALT} ({alt_dim} dims)": b.title.values}))
print("articles in both top 5s:", len(set(a.title) & set(b.title)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC | Today on Azure | In this notebook |
# MAGIC |---|---|
# MAGIC | Azure AI Search index | pgvector HNSW + Postgres full-text GIN, hybrid with RRF, **one SQL statement** |
# MAGIC | Azure OpenAI embeddings and chat | Databricks-hosted models, chosen by parameter (or keep Azure OpenAI through AI Gateway) |
# MAGIC | Custom orchestration functions | Python: this notebook, the Lakeflow Job and the app backend |
# MAGIC | A separate Postgres for AI interaction logs | `app.sessions`, `app.messages`, `app.answer_cache` in the same database |
# MAGIC | Large SQL warehouse for 4.5M-row lookups | Covering indexes in Lakebase, fed from Delta |
# MAGIC
# MAGIC **Next step we'd propose:** rerun `03_embed_and_load` and this notebook on **one real brand site's** content export,
# MAGIC and use section 6 and section 10 as the relevance and latency acceptance tests.

# COMMAND ----------

conn.close()
