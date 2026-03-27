import { memo } from "react";
import type { NodeProps } from "@xyflow/react";

function LabelNode({ data }: NodeProps) {
  const d = data as { label: string; fontSize?: number; color?: string };
  return (
    <div style={{ fontSize: d.fontSize ?? 11, fontWeight: 700, color: d.color ?? "#444", whiteSpace: "nowrap", pointerEvents: "none" }}>
      {d.label as string}
    </div>
  );
}

export default memo(LabelNode);
