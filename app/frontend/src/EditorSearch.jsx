import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api.js";
import Timings from "./Timings.jsx";

const EXAMPLES = [
  "EV charging infrastructure for depot fleets",
  "school bus driver shortage",
  "telematics ROI for small fleets",
  "nuclear verdicts insurance",
  "how do I get utility money to pay for chargers",
];

export default function EditorSearch({ facets }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [brands, setBrands] = useState([]);
  const [tags, setTags] = useState([]);
  const [dateFrom, setDateFrom] = useState("");
  const [res, setRes] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  async function run(q = query, m = mode) {
    if (!q.trim()) return;
    setQuery(q);
    setLoading(true);
    setErr(null);
    setVerdict(null);
    try {
      const r = await api.post("/api/search", {
        query: q, mode: m, brands: brands.length ? brands : null, tags: tags.length ? tags : null,
        date_from: dateFrom || null, k: 10,
      });
      setRes(r);
      if (r.results.length) {
        setVerdict({ loading: true });
        api.post("/api/search/verdict", { query: q, results: r.results })
          .then(setVerdict).catch((e) => setVerdict({ error: String(e) }));
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  const toggle = (list, set, v) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  return (
    <div className="grid2">
      <aside className="panel side">
        <h3>Search mode</h3>
        <div className="seg">
          {["hybrid", "vector", "keyword"].map((m) => (
            <button key={m} className={mode === m ? "on" : ""} onClick={() => { setMode(m); if (res) run(query, m); }}>
              {m}
            </button>
          ))}
        </div>
        <p className="hint">
          <b>hybrid</b> fuses pgvector similarity and Postgres full-text rank (RRF) in one SQL function.
        </p>
        <h3>Brands</h3>
        <div className="checks">
          {facets?.brands.map((b) => (
            <label key={b}><input type="checkbox" checked={brands.includes(b)} onChange={() => toggle(brands, setBrands, b)} /> {b}</label>
          ))}
        </div>
        <h3>Published after</h3>
        <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
        <h3>Tags</h3>
        <div className="tagcloud">
          {facets?.tags.slice(0, 24).map((t) => (
            <span key={t} className={tags.includes(t) ? "tag on" : "tag"} onClick={() => toggle(tags, setTags, t)}>{t}</span>
          ))}
        </div>
      </aside>

      <section>
        <div className="panel">
          <form className="searchbar" onSubmit={(e) => { e.preventDefault(); run(); }}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Pitch an idea: has this already been covered?" />
            <button type="submit" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
          </form>
          <div className="examples">
            {EXAMPLES.map((q) => <span key={q} className="example" onClick={() => run(q)}>{q}</span>)}
          </div>
          {res && <Timings timings={res.timings_ms} />}
        </div>

        {err && <div className="panel error">{err}</div>}

        {verdict && (
          <div className="panel verdict">
            <div className="label">Coverage check {verdict.llm_ms && <span className="muted">· {Math.round(verdict.llm_ms)} ms</span>}</div>
            {verdict.loading ? <div className="muted">Analyzing previous coverage…</div>
              : verdict.error ? <div className="error">{verdict.error}</div>
              : <ReactMarkdown>{verdict.verdict}</ReactMarkdown>}
          </div>
        )}

        {res?.results.map((r) => (
          <article key={r.article_id} className="panel result">
            <div className="rhead">
              <span className="pill">{r.brand}</span>
              <span className="muted">{r.published_at} · {r.content_type}</span>
              <span className="ranks">
                {r.vector_rank && <span className="chip">vector #{r.vector_rank}</span>}
                {r.keyword_rank && <span className="chip">keyword #{r.keyword_rank}</span>}
                <span className="chip">cos {r.cosine_sim?.toFixed(3)}</span>
              </span>
            </div>
            <h4>{r.title}</h4>
            <p className="snippet">{r.snippet}…</p>
            <div className="tagrow">{r.tags?.map((t) => <span key={t} className="tag small">{t}</span>)}</div>
          </article>
        ))}
        {res && !res.results.length && <div className="panel muted">No matches.</div>}
      </section>
    </div>
  );
}
