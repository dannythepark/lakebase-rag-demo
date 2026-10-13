import React from "react";

export default function Architecture({ health }) {
  return (
    <div className="arch">
      <div className="panel">
        <h2>Today vs. on Databricks</h2>
        <div className="compare">
          <div className="col old">
            <h3>A typical stack today</h3>
            <ul>
              <li>Azure AI Search index (separate service and bill)</li>
              <li>Azure OpenAI embeddings and LLM</li>
              <li>Custom orchestration functions holding it together</li>
              <li>A separate Postgres for app and AI interaction logs</li>
              <li>Large SQL warehouse for 4.5M-row market lookups</li>
            </ul>
          </div>
          <div className="col new">
            <h3>This demo (Databricks)</h3>
            <ul>
              <li><b>Lakebase Postgres</b>: vectors (pgvector HNSW), full-text (GIN), sessions, chat history and answer cache in <b>one database</b></li>
              <li>Databricks-hosted models, configurable (or bring Azure OpenAI through AI Gateway)</li>
              <li>Python notebooks and Lakeflow Jobs, deployed with Asset Bundles</li>
              <li>Unity Catalog Delta as the source of truth; <b>synced tables</b> feed Lakebase</li>
              <li>Autoscaling 0.5–4 CU, <b>scale to zero</b> after 5 idle minutes</li>
            </ul>
          </div>
        </div>
      </div>
      <div className="panel">
        <h3>Request path: Partner Assistant</h3>
        <ol className="flow">
          <li>Follow-up? The LLM rewrites it as a standalone question</li>
          <li>Embed the question ({health?.embedding_endpoint || "embedding endpoint"})</li>
          <li><b>Lakebase</b>: nearest cached question; if similarity ≥ 0.95, return the cached answer</li>
          <li><b>Lakebase</b>: hybrid search over campaigns and editorial chunks (RRF)</li>
          <li>Grounded answer with citations ({health?.partner_llm_endpoint || "LLM endpoint"}); rewrites and coverage checks use {health?.llm_endpoint || "a smaller model"}</li>
          <li><b>Lakebase</b>: write the message, citations, timings and cache entry</li>
        </ol>
      </div>
    </div>
  );
}
