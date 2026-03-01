import { useEffect, useState } from "react";
import type { ForestResponse } from "./types/forest";
import { fetchForest } from "./api/client";
import ForestMap from "./components/ForestMap";

export default function App() {
  const [data, setData] = useState<ForestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchForest()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", color: "#e74c3c", fontFamily: "sans-serif" }}>
        <div>
          <h2>Failed to load forest data</h2>
          <p>{error}</p>
          <p style={{ color: "#888", marginTop: 8 }}>Make sure the backend is running on port 8000.</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", fontFamily: "sans-serif" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 18, color: "#2c3e50", marginBottom: 8 }}>Loading forest...</div>
          <div style={{ color: "#888" }}>{data === null ? "Fetching from backend" : ""}</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
      <header style={{ padding: "8px 16px", background: "#2c3e50", color: "#fff", display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
        <h1 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>LightGBM Forest Visualizer</h1>
        <span style={{ fontSize: 12, color: "#bbb" }}>
          {data.num_trees} trees &middot; {data.num_rounds} rounds &middot; {data.feature_names.length} features &middot; {data.objective}
        </span>
      </header>
      <div style={{ flex: 1 }}>
        <ForestMap data={data} />
      </div>
    </div>
  );
}
