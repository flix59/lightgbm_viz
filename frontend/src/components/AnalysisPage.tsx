import { useEffect, useState } from "react";
import type { ShapSummaryResponse, PdpResponse } from "../types/forest";
import { fetchShapSummary, fetchPdp } from "../api/client";

// ---------------------------------------------------------------------------
// SVG chart dimensions
// ---------------------------------------------------------------------------
const W = 520;
const H = 260;
const PAD = { top: 25, right: 20, bottom: 40, left: 65 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function niceTicks(min: number, max: number, count: number): number[] {
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

function formatVal(v: number): string {
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  return v.toFixed(1);
}

// Color scale: low (blue) → high (red) via normalized 0–1
function valueColor(norm: number): string {
  const r = Math.round(norm * 220 + 30);
  const b = Math.round((1 - norm) * 220 + 30);
  return `rgb(${r}, 60, ${b})`;
}

// ---------------------------------------------------------------------------
// SHAP bar chart (mean |SHAP|)
// ---------------------------------------------------------------------------
function ShapBarChart({ data }: { data: ShapSummaryResponse }) {
  const top = data.features.slice(0, 10);
  const barH = 22;
  const chartH = top.length * (barH + 4) + PAD.top + PAD.bottom;
  const maxVal = Math.max(...top.map((f) => f.mean_abs_shap));

  return (
    <div>
      <h3 style={{ fontSize: 14, color: "#2c3e50", margin: "0 0 8px" }}>Feature Importance (mean |SHAP|)</h3>
      <svg width={W} height={chartH}>
        {top.map((f, i) => {
          const y = PAD.top + i * (barH + 4);
          const w = (f.mean_abs_shap / maxVal) * PW;
          return (
            <g key={f.feature_name}>
              <text x={PAD.left - 6} y={y + barH / 2 + 4} textAnchor="end" fontSize={10} fill="#555">
                {f.feature_name}
              </text>
              <rect x={PAD.left} y={y} width={w} height={barH} rx={3} fill="steelblue" opacity={0.8} />
              <text x={PAD.left + w + 4} y={y + barH / 2 + 4} fontSize={9} fill="#888">
                {f.mean_abs_shap.toFixed(0)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP beeswarm (dependence) for a single feature
// ---------------------------------------------------------------------------
function ShapBeeswarm({ data, featureIdx }: { data: ShapSummaryResponse; featureIdx: number }) {
  const f = data.features[featureIdx];
  const vals = f.values;
  const shaps = f.shap_values;

  const vMin = Math.min(...vals);
  const vMax = Math.max(...vals);
  const vRange = vMax - vMin || 1;
  const sMin = Math.min(...shaps);
  const sMax = Math.max(...shaps);
  const sRange = sMax - sMin || 1;

  const sx = (v: number) => PAD.left + ((v - vMin) / vRange) * PW;
  const sy = (s: number) => PAD.top + PH - ((s - sMin) / sRange) * PH;

  const xTicks = niceTicks(vMin, vMax, 5);
  const yTicks = niceTicks(sMin, sMax, 5);

  return (
    <div>
      <h3 style={{ fontSize: 14, color: "#2c3e50", margin: "0 0 8px" }}>SHAP Dependence: {f.feature_name}</h3>
      <svg width={W} height={H}>
        {/* Grid */}
        {yTicks.map((t) => (
          <line key={`yg${t}`} x1={PAD.left} y1={sy(t)} x2={PAD.left + PW} y2={sy(t)} stroke="#eee" />
        ))}
        {/* Zero line */}
        {sMin <= 0 && sMax >= 0 && (
          <line x1={PAD.left} y1={sy(0)} x2={PAD.left + PW} y2={sy(0)} stroke="#ccc" strokeDasharray="4 2" />
        )}
        {/* Points */}
        {vals.map((v, i) => {
          const norm = (v - vMin) / vRange;
          return <circle key={i} cx={sx(v)} cy={sy(shaps[i])} r={2.5} fill={valueColor(norm)} opacity={0.55} />;
        })}
        {/* X axis */}
        {xTicks.map((t) => (
          <g key={`xt${t}`}>
            <line x1={sx(t)} y1={PAD.top + PH} x2={sx(t)} y2={PAD.top + PH + 4} stroke="#999" />
            <text x={sx(t)} y={PAD.top + PH + 16} textAnchor="middle" fontSize={9} fill="#888">{formatVal(t)}</text>
          </g>
        ))}
        <text x={PAD.left + PW / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#666">{f.feature_name}</text>
        {/* Y axis */}
        {yTicks.map((t) => (
          <g key={`yt${t}`}>
            <line x1={PAD.left - 4} y1={sy(t)} x2={PAD.left} y2={sy(t)} stroke="#999" />
            <text x={PAD.left - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="#888">{formatVal(t)}</text>
          </g>
        ))}
        <text x={12} y={PAD.top + PH / 2} textAnchor="middle" fontSize={10} fill="#666" transform={`rotate(-90, 12, ${PAD.top + PH / 2})`}>SHAP value</text>
      </svg>
      {/* Color legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4, fontSize: 10, color: "#888" }}>
        <span>Low</span>
        <div style={{ width: 80, height: 8, background: "linear-gradient(to right, rgb(30,60,250), rgb(250,60,30))", borderRadius: 2 }} />
        <span>High</span>
        <span style={{ marginLeft: 8 }}>feature value</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PDP chart for a single feature
// ---------------------------------------------------------------------------
function PdpChart({ curve }: { curve: PdpResponse["curves"][number] }) {
  const { grid, avg_prediction, feature_values, feature_name } = curve;

  const xMin = grid[0];
  const xMax = grid[grid.length - 1];
  const yMin = Math.min(...avg_prediction);
  const yMax = Math.max(...avg_prediction);
  const yRange = yMax - yMin || 1;
  const yPad = yRange * 0.05;

  const sx = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * PW;
  const sy = (v: number) => PAD.top + PH - ((v - (yMin - yPad)) / (yRange + 2 * yPad)) * PH;

  const pathD = grid.map((g, i) => `${i === 0 ? "M" : "L"} ${sx(g)} ${sy(avg_prediction[i])}`).join(" ");

  const xTicks = niceTicks(xMin, xMax, 5);
  const yTicks = niceTicks(yMin, yMax, 5);

  return (
    <div>
      <h3 style={{ fontSize: 14, color: "#2c3e50", margin: "0 0 8px" }}>PDP: {feature_name}</h3>
      <svg width={W} height={H}>
        {/* Grid */}
        {yTicks.map((t) => (
          <line key={`yg${t}`} x1={PAD.left} y1={sy(t)} x2={PAD.left + PW} y2={sy(t)} stroke="#f0f0f0" />
        ))}
        {/* PDP line */}
        <path d={pathD} fill="none" stroke="steelblue" strokeWidth={2.5} />
        {/* Rug plot */}
        {feature_values.map((v, i) => (
          <line key={`rug${i}`} x1={sx(v)} y1={PAD.top + PH} x2={sx(v)} y2={PAD.top + PH + 4} stroke="steelblue" opacity={0.15} />
        ))}
        {/* X axis */}
        {xTicks.map((t) => (
          <g key={`xt${t}`}>
            <line x1={sx(t)} y1={PAD.top + PH} x2={sx(t)} y2={PAD.top + PH + 4} stroke="#999" />
            <text x={sx(t)} y={PAD.top + PH + 16} textAnchor="middle" fontSize={9} fill="#888">{formatVal(t)}</text>
          </g>
        ))}
        <text x={PAD.left + PW / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#666">{feature_name}</text>
        {/* Y axis */}
        {yTicks.map((t) => (
          <g key={`yt${t}`}>
            <line x1={PAD.left - 4} y1={sy(t)} x2={PAD.left} y2={sy(t)} stroke="#999" />
            <text x={PAD.left - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="#888">{formatVal(t)}</text>
          </g>
        ))}
        <text x={12} y={PAD.top + PH / 2} textAnchor="middle" fontSize={10} fill="#666" transform={`rotate(-90, 12, ${PAD.top + PH / 2})`}>Avg. prediction</text>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main analysis page
// ---------------------------------------------------------------------------
export default function AnalysisPage() {
  const [shap, setShap] = useState<ShapSummaryResponse | null>(null);
  const [pdp, setPdp] = useState<PdpResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedFeature, setSelectedFeature] = useState(0);

  useEffect(() => {
    Promise.all([fetchShapSummary(), fetchPdp()])
      .then(([s, p]) => { setShap(s); setPdp(p); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#888" }}>
        Computing SHAP values & partial dependence...
      </div>
    );
  }

  if (!shap || !pdp) return null;

  // Feature selector (ordered by SHAP importance)
  const featureNames = shap.features.map((f) => f.feature_name);

  // Find the PDP curve for the selected feature
  const selectedName = featureNames[selectedFeature];
  const pdpCurve = pdp.curves.find((c) => c.feature_name === selectedName);

  return (
    <div style={{ padding: "20px 32px", maxWidth: 1200, margin: "0 auto", overflowY: "auto", height: "100%" }}>
      <div style={{ display: "flex", gap: 40, flexWrap: "wrap" }}>
        {/* Left column: SHAP bar + feature selector */}
        <div style={{ flex: "0 0 auto" }}>
          <ShapBarChart data={shap} />
          <div style={{ marginTop: 16 }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: "#555", display: "block", marginBottom: 4 }}>
              Select feature for dependence plots:
            </label>
            <select
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(Number(e.target.value))}
              style={{ fontSize: 13, padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc" }}
            >
              {featureNames.map((name, i) => (
                <option key={name} value={i}>{name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Right column: SHAP dependence + PDP */}
        <div style={{ flex: 1, minWidth: 400 }}>
          <ShapBeeswarm data={shap} featureIdx={selectedFeature} />
          <div style={{ marginTop: 24 }}>
            {pdpCurve && <PdpChart curve={pdpCurve} />}
          </div>
        </div>
      </div>
    </div>
  );
}
