import { useEffect, useState } from "react";
import type { NodeDistributionResponse } from "../types/forest";
import { fetchNodeDistribution } from "../api/client";

interface Props {
  treeIndex: number;
  nodeId: string;
  nodeType: "split" | "leaf";
  featureName?: string;
  threshold?: number;
  leafValue?: number;
  x: number;
  y: number;
}

const W = 300;
const H = 200;
const PAD = { top: 20, right: 15, bottom: 30, left: 50 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

function ScatterPlot({ data }: { data: NodeDistributionResponse }) {
  const s = data.scatter!;
  const allX = [...s.left_x, ...s.right_x];
  const allY = [...s.left_y, ...s.right_y];
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const sx = (v: number) => PAD.left + ((v - xMin) / xRange) * PW;
  const sy = (v: number) => PAD.top + PH - ((v - yMin) / yRange) * PH;

  const thresholdX = sx(s.threshold);

  // Axis ticks
  const xTicks = [xMin, xMin + xRange / 2, xMax];
  const yTicks = [yMin, yMin + yRange / 2, yMax];

  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      {/* Axis lines */}
      <line x1={PAD.left} y1={PAD.top + PH} x2={PAD.left + PW} y2={PAD.top + PH} stroke="#ccc" />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + PH} stroke="#ccc" />

      {/* X ticks + labels */}
      {xTicks.map((v, i) => (
        <g key={`xt${i}`}>
          <line x1={sx(v)} y1={PAD.top + PH} x2={sx(v)} y2={PAD.top + PH + 4} stroke="#999" />
          <text x={sx(v)} y={PAD.top + PH + 14} textAnchor="middle" fontSize={8} fill="#888">
            {v.toFixed(0)}
          </text>
        </g>
      ))}

      {/* Y ticks + labels */}
      {yTicks.map((v, i) => (
        <g key={`yt${i}`}>
          <line x1={PAD.left - 4} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#999" />
          <text x={PAD.left - 6} y={sy(v) + 3} textAnchor="end" fontSize={8} fill="#888">
            ${(v / 1000).toFixed(0)}k
          </text>
        </g>
      ))}

      {/* Axis labels */}
      <text x={PAD.left + PW / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="#666">
        {s.feature_name}
      </text>
      <text x={10} y={PAD.top + PH / 2} textAnchor="middle" fontSize={9} fill="#666" transform={`rotate(-90, 10, ${PAD.top + PH / 2})`}>
        Price
      </text>

      {/* Left points (≤ threshold) */}
      {s.left_x.map((xv, i) => (
        <circle key={`l${i}`} cx={sx(xv)} cy={sy(s.left_y[i])} r={2} fill="steelblue" opacity={0.45} />
      ))}

      {/* Right points (> threshold) */}
      {s.right_x.map((xv, i) => (
        <circle key={`r${i}`} cx={sx(xv)} cy={sy(s.right_y[i])} r={2} fill="tomato" opacity={0.45} />
      ))}

      {/* Threshold line */}
      <line x1={thresholdX} y1={PAD.top} x2={thresholdX} y2={PAD.top + PH} stroke="#333" strokeWidth={1.5} strokeDasharray="4 2" />
      <text x={thresholdX + 3} y={PAD.top + 10} fontSize={8} fill="#333" fontWeight={600}>
        {s.threshold.toFixed(2)}
      </text>
    </svg>
  );
}

function Histogram({ data }: { data: NodeDistributionResponse }) {
  const maxCount = Math.max(...data.histogram.counts);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", height: 50, gap: 1 }}>
      {data.histogram.counts.map((count, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            height: `${(count / maxCount) * 100}%`,
            background: "#58d68d",
            borderRadius: "1px 1px 0 0",
            minHeight: count > 0 ? 2 : 0,
          }}
        />
      ))}
    </div>
  );
}

export default function LeafTooltip({ treeIndex, nodeId, nodeType, featureName, threshold, leafValue, x, y }: Props) {
  const [data, setData] = useState<NodeDistributionResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      fetchNodeDistribution(treeIndex, nodeId).then((d) => {
        if (!cancelled) setData(d);
      });
    }, 150);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [treeIndex, nodeId]);

  // Position tooltip so it doesn't overflow viewport
  const tooltipWidth = data?.scatter ? W + 32 : 260;
  const left = x + tooltipWidth + 20 > window.innerWidth ? x - tooltipWidth - 12 : x + 12;
  const top = Math.max(10, Math.min(y - 10, window.innerHeight - 340));

  if (!data) {
    return (
      <div style={{ position: "fixed", left, top, zIndex: 1000, background: "#fff", border: "1px solid #ccc", borderRadius: 8, padding: "10px 14px", boxShadow: "0 4px 12px rgba(0,0,0,0.15)", fontSize: 12 }}>
        Loading...
      </div>
    );
  }

  const totalSamples = data.histogram.counts.reduce((a, b) => a + b, 0);
  const { target_stats: stats } = data;

  const title = nodeType === "split"
    ? `${featureName} ≤ ${threshold?.toFixed(2)}`
    : `Leaf value: ${leafValue?.toFixed(3)}`;

  return (
    <div style={{ position: "fixed", left, top, zIndex: 1000, background: "#fff", border: "1px solid #ccc", borderRadius: 8, padding: "12px 16px", boxShadow: "0 4px 12px rgba(0,0,0,0.15)", fontSize: 12 }}>
      <div style={{ fontWeight: 700, marginBottom: 2, color: "#2c3e50" }}>{title}</div>
      <div style={{ color: "#888", marginBottom: 6 }}>
        n={totalSamples}
        {data.scatter && (
          <span>
            {" "}&middot;{" "}
            <span style={{ color: "steelblue", fontWeight: 600 }}>&#9679; ≤ ({data.scatter.left_x.length})</span>
            {" "}
            <span style={{ color: "tomato", fontWeight: 600 }}>&#9679; &gt; ({data.scatter.right_x.length})</span>
          </span>
        )}
      </div>

      {data.scatter ? <ScatterPlot data={data} /> : <Histogram data={data} />}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 12px", color: "#555", marginTop: 6 }}>
        <span>Mean: <b>${stats.mean.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
        <span>Median: <b>${stats.median.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
        <span>Std: <b>${stats.std.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
        <span>Q25–Q75: <b>${stats.q25.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b>–<b>${stats.q75.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
      </div>
    </div>
  );
}
