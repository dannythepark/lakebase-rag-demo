import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import EditorSearch from "./EditorSearch.jsx";
import PartnerAssistant from "./PartnerAssistant.jsx";
import MarketLookup from "./MarketLookup.jsx";
import Architecture from "./Architecture.jsx";

const TABS = [
  { id: "search", label: "Editor Search", sub: "internal" },
  { id: "partner", label: "Partner Assistant", sub: "external" },
  { id: "market", label: "Market Lookup", sub: "4.5M records" },
  { id: "arch", label: "How it works", sub: "" },
];

export default function App() {
  const [tab, setTab] = useState("search");
  const [facets, setFacets] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.get("/api/facets").then(setFacets).catch(() => setFacets(null));
    api.get("/api/health").then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo">◆</span>
          <div>
            <div className="title">Content Assistant</div>
            <div className="subtitle">One Lakebase Postgres · pgvector + full-text · Databricks-hosted models</div>
          </div>
        </div>
        {facets?.stats && (
          <div className="stats">
            <Stat n={facets.stats.articles} label="articles" />
            <Stat n={facets.stats.chunks} label="chunks" />
            <Stat n={facets.stats.campaigns} label="campaigns" />
            <Stat n={facets.brands.length} label="brands" />
          </div>
        )}
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
            {t.label}
            {t.sub && <span className="tabsub">{t.sub}</span>}
          </button>
        ))}
      </nav>
      <main>
        {tab === "search" && <EditorSearch facets={facets} />}
        {tab === "partner" && <PartnerAssistant />}
        {tab === "market" && <MarketLookup />}
        {tab === "arch" && <Architecture health={health} />}
      </main>
    </div>
  );
}

function Stat({ n, label }) {
  return (
    <div className="stat">
      <div className="statn">{Number(n).toLocaleString()}</div>
      <div className="statl">{label}</div>
    </div>
  );
}
