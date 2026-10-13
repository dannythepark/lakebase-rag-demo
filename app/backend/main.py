"""Content Assistant: Editor Search + Partner Assistant + Market Lookup on Lakebase."""
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from .clients import EMBEDDING_ENDPOINT, LLM_ENDPOINT, PARTNER_LLM_ENDPOINT, chat, embed, vec_literal
from .db import pool

CACHE_SIMILARITY = 0.95
STATIC = Path(__file__).resolve().parent.parent / "static"

APP_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS app;
CREATE TABLE IF NOT EXISTS app.sessions (
  session_id  UUID PRIMARY KEY,
  title       TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app.messages (
  message_id  BIGSERIAL PRIMARY KEY,
  session_id  UUID NOT NULL REFERENCES app.sessions (session_id) ON DELETE CASCADE,
  role        TEXT NOT NULL,
  content     TEXT NOT NULL,
  citations   JSONB,
  meta        JSONB,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session ON app.messages (session_id, message_id);
CREATE TABLE IF NOT EXISTS app.answer_cache (
  cache_id            BIGSERIAL PRIMARY KEY,
  question            TEXT NOT NULL,
  question_embedding  VECTOR(1024) NOT NULL,
  answer              TEXT NOT NULL,
  citations           JSONB,
  hit_count           INT NOT NULL DEFAULT 0,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_hit_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS answer_cache_hnsw ON app.answer_cache USING hnsw (question_embedding vector_cosine_ops);
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pool.open(wait=True, timeout=60)
    with pool.connection() as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            pass  # already created by the project owner
        if conn.execute("SELECT to_regclass('app.answer_cache')").fetchone()[0] is None:
            conn.execute(APP_SCHEMA)  # first start: the app's service principal creates and owns `app`
    yield
    pool.close()


app = FastAPI(title="Content Assistant", lifespan=lifespan)


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# ---------------------------------------------------------------- meta


@app.get("/api/health")
def health():
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True, "embedding_endpoint": EMBEDDING_ENDPOINT, "llm_endpoint": LLM_ENDPOINT,
            "partner_llm_endpoint": PARTNER_LLM_ENDPOINT}


@app.get("/api/facets")
def facets():
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        brands = [r["brand"] for r in cur.execute("SELECT DISTINCT brand FROM content.chunks ORDER BY 1")]
        types = [r["content_type"] for r in cur.execute("SELECT DISTINCT content_type FROM content.chunks ORDER BY 1")]
        tags = [
            r["tag"]
            for r in cur.execute(
                "SELECT tag FROM (SELECT unnest(tags) tag FROM content.chunks) t GROUP BY tag ORDER BY count(*) DESC LIMIT 40"
            )
        ]
        stats = cur.execute(
            "SELECT count(DISTINCT article_id) AS articles, count(*) AS chunks, min(published_at) AS min_date, "
            "max(published_at) AS max_date FROM content.chunks"
        ).fetchone()
        stats["campaigns"] = cur.execute("SELECT count(*) AS n FROM content.campaigns").fetchone()["n"]
    return {"brands": brands, "content_types": types, "tags": tags, "stats": stats}


# ---------------------------------------------------------------- editor search


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    mode: str = "hybrid"
    brands: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    content_types: Optional[list[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    k: int = 10


@app.post("/api/search")
def search(req: SearchRequest):
    if req.mode not in ("hybrid", "vector", "keyword"):
        raise HTTPException(400, "mode must be hybrid, vector or keyword")
    t0 = time.perf_counter()
    q_emb = vec_literal(embed(req.query))
    embed_ms = ms_since(t0)

    t1 = time.perf_counter()
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """SELECT * FROM content.search_chunks(
                 %(q)s, %(e)s::vector, k => %(n)s, mode => %(mode)s,
                 brand_filter => %(brands)s, tag_filter => %(tags)s, type_filter => %(types)s,
                 date_from => %(df)s, date_to => %(dt)s)""",
            dict(q=req.query, e=q_emb, n=req.k * 4, mode=req.mode, brands=req.brands or None,
                 tags=req.tags or None, types=req.content_types or None, df=req.date_from, dt=req.date_to),
        ).fetchall()
    db_ms = ms_since(t1)

    # collapse chunks to articles, keeping the best-scoring chunk as the snippet
    articles: dict[str, dict] = {}
    for r in rows:
        a = articles.get(r["article_id"])
        if a is None:
            articles[r["article_id"]] = a = {**r, "matching_chunks": 0}
            a.pop("chunk_text")
            a["snippet"] = r["chunk_text"][:420]
        a["matching_chunks"] += 1
    results = list(articles.values())[: req.k]
    return {
        "query": req.query,
        "mode": req.mode,
        "results": results,
        "timings_ms": {"embed": embed_ms, "lakebase_search": db_ms, "total": ms_since(t0)},
    }


class VerdictRequest(BaseModel):
    query: str
    results: list[dict]


@app.post("/api/search/verdict")
def verdict(req: VerdictRequest):
    t0 = time.perf_counter()
    listing = "\n".join(
        f"- [{r.get('brand')}, {r.get('published_at')}, {r.get('content_type')}] {r.get('title')} :: {r.get('snippet', '')[:300]}"
        for r in req.results[:8]
    )
    text = chat(
        [
            {
                "role": "system",
                "content": "You help B2B trade-media editors avoid duplicate coverage. Be concise and specific. "
                "Only use the search results provided.",
            },
            {
                "role": "user",
                "content": f'An editor is pitching: "{req.query}"\n\nMost relevant previously published pieces:\n{listing}\n\n'
                "In at most 90 words: (1) say whether this is already well covered, how often, by which brands and "
                "how recently; (2) suggest 2 fresh angles not yet covered. Format: one short paragraph, then "
                '"Fresh angles:" followed by two bullet points.',
            },
        ],
        max_tokens=300,
    )
    return {"verdict": text, "llm_ms": ms_since(t0)}


# ---------------------------------------------------------------- partner assistant

PARTNER_SYSTEM = """You are the partnerships assistant for a B2B trade-media company that publishes eight fleet and
transportation brands (FleetPulse, Upfit & Haul, Transit Forward, Yellow Route, Dealer Lot Report, Bay & Bench,
Charge Depot, SafeMiles). You talk to prospective advertisers and sponsors.

Rules:
- Answer ONLY from the numbered context. Cite sources inline like [1], [2].
- Recommend specific offerings (package, brands, price range) that fit the advertiser's goal and audience.
- Use past campaign case studies as proof points when relevant.
- Never invent prices, audience numbers or results that aren't in the context.
- For any price comparison ("cheapest", "lower price point", "entry price"), use ONLY the "Entry prices" list at the
  end of the context, which is already sorted from cheapest to most expensive. Never reorder it. If you call any
  option the cheapest, lowest-priced or most affordable, it must be item 1 on that list.
- If the context doesn't cover the question, say so and offer to connect them with the sales team.
- Keep answers under 200 words. Friendly, professional, no fluff."""


class NewSession(BaseModel):
    title: Optional[str] = None


@app.post("/api/partner/sessions")
def create_session(req: NewSession):
    sid = uuid.uuid4()
    with pool.connection() as conn:
        conn.execute("INSERT INTO app.sessions (session_id, title) VALUES (%s, %s)", (sid, req.title))
    return {"session_id": str(sid)}


@app.get("/api/partner/sessions")
def list_sessions():
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT s.session_id, coalesce(s.title, 'New conversation') AS title, s.updated_at,
                      (SELECT count(*) FROM app.messages m WHERE m.session_id = s.session_id) AS n_messages
               FROM app.sessions s ORDER BY s.updated_at DESC LIMIT 25"""
        ).fetchall()


@app.get("/api/partner/sessions/{session_id}")
def get_session(session_id: uuid.UUID):
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        msgs = cur.execute(
            "SELECT role, content, citations, meta, created_at FROM app.messages WHERE session_id = %s ORDER BY message_id",
            (session_id,),
        ).fetchall()
    return {"session_id": str(session_id), "messages": msgs}


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    message: str = Field(..., min_length=1)
    use_cache: bool = True


def _entry_price(price_range: Optional[str]) -> tuple[float, str]:
    """'18,000-35,000 / month' -> (18000.0, '/ month'). Unparseable ranges sort last."""
    if not price_range:
        return float("inf"), ""
    low, _, rest = price_range.partition("-")
    unit = rest.partition("/")[2].strip()
    try:
        return float(low.replace(",", "").replace("$", "").strip()), f"/ {unit}" if unit else ""
    except ValueError:
        return float("inf"), ""


def _standalone_question(history: list[dict], message: str) -> str:
    """Rewrite a follow-up into a self-contained question so retrieval and caching work across turns."""
    convo = "\n".join(f"{m['role']}: {m['content'][:2000]}" for m in history[-6:])
    return chat(
        [
            {"role": "system", "content": "Rewrite the user's last message as a single standalone question that "
             "keeps every relevant detail from the conversation (the advertiser's product, target audience, "
             "brands, budget, and which options 'those'/'that' refer to by name). Never answer the question "
             "or include claims from earlier answers, such as which option is cheapest or best. Output only the question."},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {message}"},
        ],
        max_tokens=80,
        temperature=0,
    ).strip()


@app.post("/api/partner/chat")
def partner_chat(req: ChatRequest):
    t0 = time.perf_counter()
    timings = {}
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if not cur.execute("SELECT 1 FROM app.sessions WHERE session_id = %s", (req.session_id,)).fetchone():
            raise HTTPException(404, "unknown session")
        history = cur.execute(
            "SELECT role, content, citations FROM app.messages WHERE session_id = %s ORDER BY message_id",
            (req.session_id,),
        ).fetchall()
    # campaigns cited in the last answer stay in context, so "which of those…" follow-ups can compare them
    prior_ids = [c["id"] for m in history[-1:] if m["role"] == "assistant" for c in (m["citations"] or [])
                 if c.get("kind") != "article"]

    question = req.message
    if history:
        t = time.perf_counter()
        question = _standalone_question(history, req.message)
        timings["rewrite_llm"] = ms_since(t)

    t = time.perf_counter()
    q_emb = vec_literal(embed(question))
    timings["embed"] = ms_since(t)

    cached, entry_prices = None, None
    if req.use_cache:
        t = time.perf_counter()
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cached = cur.execute(
                """SELECT cache_id, question, answer, citations, 1 - (question_embedding <=> %s::vector) AS sim
                   FROM app.answer_cache ORDER BY question_embedding <=> %s::vector LIMIT 1""",
                (q_emb, q_emb),
            ).fetchone()
        timings["cache_lookup"] = ms_since(t)
        if cached and cached["sim"] < CACHE_SIMILARITY:
            cached = None

    if cached:
        answer, citations = cached["answer"], cached["citations"]
        with pool.connection() as conn:
            conn.execute(
                "UPDATE app.answer_cache SET hit_count = hit_count + 1, last_hit_at = now() WHERE cache_id = %s",
                (cached["cache_id"],),
            )
        cache_info = {"hit": True, "similarity": round(cached["sim"], 4), "cached_question": cached["question"]}
    else:
        t = time.perf_counter()
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            camps = cur.execute(
                "SELECT * FROM content.search_campaigns(%s, %s::vector, k => 6)", (question, q_emb)
            ).fetchall()
            found = {c["campaign_id"] for c in camps}
            missing = [i for i in prior_ids if i not in found]
            if missing:
                camps = cur.execute(
                    """SELECT campaign_id, record_type, name, package_type, sponsor_category, brands, target_audience,
                              price_tier, price_range_usd, year, description
                       FROM content.campaigns WHERE campaign_id = ANY(%s)""",
                    (missing,),
                ).fetchall() + camps
            chunks = cur.execute(
                "SELECT * FROM content.search_chunks(%s, %s::vector, k => 12)", (question, q_emb)
            ).fetchall()
        timings["lakebase_retrieval"] = ms_since(t)

        citations, context = [], []
        for c in camps:
            n = len(citations) + 1
            label = "Media kit offering" if c["record_type"] == "offering" else f"Past campaign ({c['year']})"
            citations.append({"n": n, "kind": c["record_type"], "id": c["campaign_id"], "title": c["name"],
                              "brands": c["brands"], "price_range_usd": c["price_range_usd"]})
            context.append(
                f"[{n}] {label}: {c['name']} | package: {c['package_type']} | brands: {', '.join(c['brands'] or [])} | "
                f"price: ${c['price_range_usd']} | audience: {c['target_audience']}\n{c['description']}"
            )
        # computed in code so the LLM never has to compare numbers itself
        packages: dict[str, dict] = {}
        for i, c in enumerate(camps, 1):
            low, unit = _entry_price(c["price_range_usd"])
            if low == float("inf"):
                continue
            pkg = packages.setdefault(c["package_type"] or c["name"], {"low": low, "unit": unit, "refs": []})
            if low < pkg["low"]:
                pkg.update(low=low, unit=unit)
            pkg["refs"].append(f"[{i}]")
        entry_prices = "\n".join(
            f"{rank}. {name}: from ${p['low']:,.0f} {p['unit']}".rstrip() + f" {''.join(p['refs'])}"
            for rank, (name, p) in enumerate(sorted(packages.items(), key=lambda kv: kv[1]["low"]), 1)
        )
        seen = set()
        for ch in chunks:
            if ch["article_id"] in seen or len(seen) >= 4:
                continue
            seen.add(ch["article_id"])
            n = len(citations) + 1
            citations.append({"n": n, "kind": "article", "id": ch["article_id"], "title": ch["title"],
                              "brands": [ch["brand"]], "url": ch["url"], "published_at": str(ch["published_at"])})
            context.append(f"[{n}] Editorial coverage: {ch['brand']} ({ch['published_at']}): {ch['title']}\n{ch['chunk_text'][:900]}")

        t = time.perf_counter()
        msgs = [{"role": "system", "content": PARTNER_SYSTEM}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        if entry_prices:
            context.append(f"Entry prices by package type, cheapest first (computed from the records above):\n{entry_prices}")
        msgs.append({"role": "user", "content": f"Context:\n\n" + "\n\n".join(context) + f"\n\nQuestion: {question}"})
        answer = chat(msgs, max_tokens=500, endpoint=PARTNER_LLM_ENDPOINT)
        timings["answer_llm"] = ms_since(t)

        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO app.answer_cache (question, question_embedding, answer, citations) VALUES (%s, %s::vector, %s, %s)",
                (question, q_emb, answer, json.dumps(citations)),
            )
        cache_info = {"hit": False}

    timings["total"] = ms_since(t0)
    meta = {"standalone_question": question, "cache": cache_info, "timings_ms": timings}
    if entry_prices:
        meta["entry_prices"] = entry_prices
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO app.messages (session_id, role, content) VALUES (%s, 'user', %s)", (req.session_id, req.message)
        )
        conn.execute(
            "INSERT INTO app.messages (session_id, role, content, citations, meta) VALUES (%s, 'assistant', %s, %s, %s)",
            (req.session_id, answer, json.dumps(citations), json.dumps(meta)),
        )
        conn.execute(
            "UPDATE app.sessions SET updated_at = now(), title = coalesce(title, left(%s, 60)) WHERE session_id = %s",
            (req.message, req.session_id),
        )
    return {"answer": answer, "citations": citations, **meta}


@app.get("/api/partner/cache")
def cache_stats():
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT count(*) AS entries, coalesce(sum(hit_count), 0) AS hits FROM app.answer_cache"
        ).fetchone()


# ---------------------------------------------------------------- market lookup (synced table)


@app.get("/api/market/facets")
def market_facets():
    try:
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            states = [r["state"] for r in cur.execute("SELECT DISTINCT state FROM market.companies ORDER BY 1")]
            verticals = [r["vertical"] for r in cur.execute("SELECT DISTINCT vertical FROM market.companies ORDER BY 1")]
            total = cur.execute(
                "SELECT reltuples::bigint AS n FROM pg_class WHERE oid = 'market.companies'::regclass"
            ).fetchone()["n"]
    except Exception as e:
        raise HTTPException(404, f"market data not loaded yet: {e}")
    return {"states": states, "verticals": verticals, "approx_rows": total}


class MarketQuery(BaseModel):
    state: Optional[str] = None
    vertical: Optional[str] = None
    min_fleet: int = 1
    max_fleet: int = 100000
    telematics_only: bool = False


@app.post("/api/market/lookup")
def market_lookup(req: MarketQuery):
    t0 = time.perf_counter()
    # only include the filters that are set, so Postgres plans each shape on the covering indexes
    clauses = ["fleet_size BETWEEN %(lo)s AND %(hi)s"]
    if req.state:
        clauses.append("state = %(state)s")
    if req.vertical:
        clauses.append("vertical = %(vertical)s")
    if req.telematics_only:
        clauses.append("has_telematics")
    where = " AND ".join(clauses)
    params = dict(state=req.state, vertical=req.vertical, lo=req.min_fleet, hi=req.max_fleet)
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        summary = cur.execute(
            f"""SELECT count(*) AS companies, coalesce(sum(fleet_size), 0) AS vehicles,
                       coalesce(sum(ev_count), 0) AS evs,
                       round(100.0 * avg(has_telematics::int), 1) AS pct_telematics,
                       round(avg(fleet_size), 1) AS avg_fleet
                FROM market.companies WHERE {where}""",
            params,
        ).fetchone()
        top = cur.execute(
            f"""SELECT company_name, state, vertical, fleet_size, ev_count, has_telematics
                FROM market.companies WHERE {where} ORDER BY fleet_size DESC LIMIT 10""",
            params,
        ).fetchall()
    return {"summary": summary, "top": top, "lakebase_ms": ms_since(t0)}


# ---------------------------------------------------------------- frontend

if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = STATIC / path
        return FileResponse(f if path and f.is_file() else STATIC / "index.html")
