import React from "react";

export default function Timings({ timings }) {
  if (!timings) return null;
  return (
    <div className="timings">
      {Object.entries(timings).map(([k, v]) => (
        <span key={k} className={k === "total" ? "chip total" : k.startsWith("lakebase") || k === "cache_lookup" ? "chip lb" : "chip"}>
          {k.replaceAll("_", " ")}: <b>{Math.round(v)} ms</b>
        </span>
      ))}
    </div>
  );
}
