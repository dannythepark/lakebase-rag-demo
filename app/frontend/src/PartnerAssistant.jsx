import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api.js";
import Timings from "./Timings.jsx";

const STARTERS = [
  "We sell AI dash cams to mid-size fleets. What sponsorship options reach fleet safety managers?",
  "We're an EV charging company launching a depot product. How could we partner with you?",
  "What results have telematics vendors seen from past campaigns?",
  "Do you have webinar sponsorships for school transportation directors, and what do they cost?",
];

export default function PartnerAssistant() {
  const [sessions, setSessions] = useState([]);
  const [sid, setSid] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [cache, setCache] = useState(null);
  const endRef = useRef(null);

  const refresh = () => {
    api.get("/api/partner/sessions").then(setSessions).catch(() => {});
    api.get("/api/partner/cache").then(setCache).catch(() => {});
  };
  useEffect(refresh, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function open(id) {
    setSid(id);
    const s = await api.get(`/api/partner/sessions/${id}`);
    setMessages(s.messages);
  }

  async function newSession() {
    const s = await api.post("/api/partner/sessions", {});
    setSid(s.session_id);
    setMessages([]);
    return s.session_id;
  }

  async function send(text = input) {
    if (!text.trim() || busy) return;
    setBusy(true);
    setErr(null);
    setInput("");
    const id = sid || (await newSession());
    setMessages((m) => [...m, { role: "user", content: text }]);
    try {
      const r = await api.post("/api/partner/chat", { session_id: id, message: text });
      setMessages((m) => [...m, { role: "assistant", content: r.answer, citations: r.citations,
        meta: { cache: r.cache, timings_ms: r.timings_ms, standalone_question: r.standalone_question } }]);
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid2">
      <aside className="panel side">
        <button className="primary wide" onClick={newSession}>+ New conversation</button>
        <h3>Sessions <span className="muted">(stored in Lakebase)</span></h3>
        <div className="sessions">
          {sessions.map((s) => (
            <div key={s.session_id} className={s.session_id === sid ? "session on" : "session"} onClick={() => open(s.session_id)}>
              <div className="stitle">{s.title}</div>
              <div className="muted small">{s.n_messages} messages</div>
            </div>
          ))}
        </div>
        {cache && (
          <div className="cachebox">
            <h3>Semantic answer cache</h3>
            <div><b>{cache.entries}</b> cached answers · <b>{cache.hits}</b> hits</div>
            <p className="hint">Questions with cosine similarity ≥ 0.95 to a cached question skip retrieval and the LLM.</p>
          </div>
        )}
      </aside>

      <section className="panel chat">
        <div className="chatlog">
          {!messages.length && (
            <div className="welcome">
              <h2>Partner with our fleet &amp; transportation brands</h2>
              <p className="muted">Ask about audiences, sponsorships, and results from past campaigns.</p>
              <div className="starters">
                {STARTERS.map((s) => <div key={s} className="starter" onClick={() => send(s)}>{s}</div>)}
              </div>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} m={m} />)}
          {busy && <div className="msg assistant"><div className="bubble muted">Thinking…</div></div>}
          {err && <div className="error">{err}</div>}
          <div ref={endRef} />
        </div>
        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about sponsorship options…" disabled={busy} />
          <button type="submit" disabled={busy}>Send</button>
        </form>
      </section>
    </div>
  );
}

function Message({ m }) {
  const meta = m.meta;
  return (
    <div className={`msg ${m.role}`}>
      <div className="bubble">
        {m.role === "assistant" ? <ReactMarkdown>{m.content}</ReactMarkdown> : m.content}
      </div>
      {m.role === "assistant" && meta && (
        <div className="msgmeta">
          {meta.cache?.hit
            ? <span className="chip hit">⚡ cache hit · sim {meta.cache.similarity}</span>
            : <span className="chip">fresh answer</span>}
          {meta.standalone_question && meta.standalone_question !== "" && (
            <span className="muted small">retrieval query: “{meta.standalone_question}”</span>
          )}
          <Timings timings={meta.timings_ms} />
        </div>
      )}
      {m.role === "assistant" && m.citations?.length > 0 && (
        <details className="cites">
          <summary>{m.citations.length} sources</summary>
          {m.citations.map((c) => (
            <div key={c.n} className="cite">
              <b>[{c.n}]</b> <span className="pill small">{c.kind.replace("_", " ")}</span> {c.title}
              <span className="muted small"> · {c.brands?.join(", ")}{c.price_range_usd ? ` · $${c.price_range_usd}` : ""}{c.published_at ? ` · ${c.published_at}` : ""}</span>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
