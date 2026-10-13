import React, { useEffect, useState } from "react";
import { api } from "./api.js";

export default function MarketLookup() {
  const [facets, setFacets] = useState(null);
  const [err, setErr] = useState(null);
  const [q, setQ] = useState({ state: "TX", vertical: "", min_fleet: 50, max_fleet: 200, telematics_only: false });
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get("/api/market/facets").then(setFacets).catch((e) => setErr(String(e)));
  }, []);

  async function run(e) {
    e?.preventDefault();
    setBusy(true);
    try {
      setRes(await api.post("/api/market/lookup", { ...q, state: q.state || null, vertical: q.vertical || null,
        min_fleet: Number(q.min_fleet), max_fleet: Number(q.max_fleet) }));
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const set = (k) => (e) => setQ({ ...q, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  if (err && !facets) return <div className="panel muted">Market data isn't loaded yet. Run the <code>market_intel</code> job. ({err})</div>;

  return (
    <div>
      <div className="panel">
        <p className="muted">
          ~{facets ? Number(facets.approx_rows).toLocaleString() : "…"} company/fleet records. Source of truth is Delta in
          Unity Catalog, loaded into <b>Lakebase</b> (a synced table in production; bulk <code>COPY</code> in this demo workspace)
          and served with covering B-tree indexes. No SQL warehouse is involved in these lookups.
        </p>
        <form className="marketform" onSubmit={run}>
          <label>State
            <select value={q.state} onChange={set("state")}>
              <option value="">Any</option>
              {facets?.states.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label>Vertical
            <select value={q.vertical} onChange={set("vertical")}>
              <option value="">Any</option>
              {facets?.verticals.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label>Fleet size from <input type="number" value={q.min_fleet} onChange={set("min_fleet")} /></label>
          <label>to <input type="number" value={q.max_fleet} onChange={set("max_fleet")} /></label>
          <label className="inline"><input type="checkbox" checked={q.telematics_only} onChange={set("telematics_only")} /> uses telematics</label>
          <button type="submit" disabled={busy}>{busy ? "…" : "Look up"}</button>
        </form>
      </div>
      {err && facets && <div className="panel error">{err}</div>}
      {res && (
        <>
          <div className="kpis">
            <Kpi label="companies" v={res.summary.companies} />
            <Kpi label="vehicles" v={res.summary.vehicles} />
            <Kpi label="EVs" v={res.summary.evs} />
            <Kpi label="% with telematics" v={res.summary.pct_telematics} />
            <Kpi label="Lakebase query" v={`${Math.round(res.lakebase_ms)} ms`} accent />
          </div>
          <div className="panel">
            <h3>Largest fleets matching</h3>
            <table>
              <thead><tr><th>Company</th><th>State</th><th>Vertical</th><th>Fleet</th><th>EVs</th><th>Telematics</th></tr></thead>
              <tbody>
                {res.top.map((r) => (
                  <tr key={r.company_name}><td>{r.company_name}</td><td>{r.state}</td><td>{r.vertical}</td>
                    <td>{r.fleet_size.toLocaleString()}</td><td>{r.ev_count}</td><td>{r.has_telematics ? "yes" : "no"}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function Kpi({ label, v, accent }) {
  return (
    <div className={accent ? "kpi accent" : "kpi"}>
      <div className="kpiv">{typeof v === "number" ? v.toLocaleString() : v}</div>
      <div className="kpil">{label}</div>
    </div>
  );
}
