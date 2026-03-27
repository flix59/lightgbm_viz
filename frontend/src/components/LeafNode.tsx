import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

interface LeafNodeData {
  leafValue: number;
  sampleCount: number;
  width: number;
  height: number;
  scale: number;
  [key: string]: unknown;
}

function LeafNode({ data }: NodeProps) {
  const d = data as unknown as LeafNodeData;
  const fontSize = Math.max(8, 9 * d.scale);

  return (
    <div
      style={{
        width: d.width,
        height: d.height,
        background: "#d5f5e3",
        border: "1.5px solid #58d68d",
        borderRadius: 6,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 3,
        overflow: "hidden",
        cursor: "pointer",
      }}
    >
      <div style={{ fontSize, fontWeight: 600, color: "#1e8449", lineHeight: 1.2 }}>
        {d.sampleCount}
      </div>
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 0, height: 0, minWidth: 0, minHeight: 0, border: "none" }} />
    </div>
  );
}

export default memo(LeafNode);
