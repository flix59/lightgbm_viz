import { useEffect, useState } from "react";
import type { ForestResponse } from "./types/forest";
import { fetchForest, fetchSamples, type SamplesData } from "./api/client";
import ForestMap from "./components/ForestMap";
import AnalysisPage from "./components/AnalysisPage";

type Page = "forest" | "analysis";

export default function App() {
  const [data, setData] = useState<ForestResponse | null>(null);
  const [samples, setSamples] = useState<SamplesData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("forest");

  useEffect(() => {
    Promise.all([fetchForest(), fetchSamples()])
      .then(([f, s]) => { setData(f); setSamples(s); })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", color: "#e74c3c", fontFamily: "sans-serif" }}>
        <div>
          <h2>Failed to load data</h2>
          <p>{error}</p>
          <p style={{ color: "#888", marginTop: 8 }}>Run <code>uv run precompute.py</code> first to generate the data files.</p>
        </div>
      </div>
    );
  }

  if (!data || !samples) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", fontFamily: "sans-serif" }}>
        <div style={{ textAlign: "center", fontSize: 18, color: "#2c3e50" }}>Loading forest...</div>
      </div>
    );
  }

  const tabStyle = (active: boolean) => ({
    padding: "6px 16px",
    fontSize: 13,
    fontWeight: active ? 700 : 400,
    color: active ? "#fff" : "#bbb",
    background: active ? "rgba(255,255,255,0.15)" : "transparent",
    border: "none",
    borderRadius: 4,
    cursor: "pointer" as const,
  });

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
      <header style={{ padding: "6px 16px", background: "#2c3e50", color: "#fff", display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
        <h1 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>LightGBM Forest Visualizer</h1>
        <span style={{ fontSize: 12, color: "#bbb" }}>
          {data.num_trees} trees &middot; {data.num_rounds} rounds &middot; {data.feature_names.length} features &middot; {data.objective}
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          <button style={tabStyle(page === "forest")} onClick={() => setPage("forest")}>Forest Map</button>
          <button style={tabStyle(page === "analysis")} onClick={() => setPage("analysis")}>SHAP & PDP</button>
        </div>
      </header>
      <div style={{ flex: 1, overflow: "hidden" }}>
        {page === "forest" ? <ForestMap data={data} samples={samples} /> : <AnalysisPage />}
      </div>
    </div>
  );
}
