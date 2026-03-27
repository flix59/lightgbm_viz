import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

interface SplitNodeData {
  featureName: string;
  threshold: number;
  sampleCount: number;
  width: number;
  height: number;
  scale: number;
  [key: string]: unknown;
}

function SplitNode({ data }: NodeProps) {
  const d = data as unknown as SplitNodeData;
  const fontSize = Math.max(8, 9 * d.scale);

  return (
    <div
      style={{
        width: d.width,
        height: d.height,
        background: "#d4e6f1",
        border: "1.5px solid #5b9bd5",
        borderRadius: 6,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 3,
        overflow: "hidden",
      }}
    >
      <div style={{ fontSize, fontWeight: 600, color: "#2c3e50", lineHeight: 1.2, textAlign: "center", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%" }}>
        {d.featureName}
      </div>
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 0, height: 0, minWidth: 0, minHeight: 0, border: "none" }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 0, height: 0, minWidth: 0, minHeight: 0, border: "none" }} />
    </div>
  );
}

export default memo(SplitNode);
