import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  MiniMap,
  Background,
  useReactFlow,
  ReactFlowProvider,
  type NodeMouseHandler,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { ForestResponse } from "../types/forest";
import { layoutForest } from "../layout/forestLayout";
import SplitNode from "./SplitNode";
import LeafNode from "./LeafNode";
import LeafTooltip from "./LeafTooltip";

const nodeTypes = { splitNode: SplitNode, leafNode: LeafNode };

interface Props {
  data: ForestResponse;
}

interface HoverInfo {
  treeIndex: number;
  nodeId: string;
  nodeType: "split" | "leaf";
  featureName?: string;
  threshold?: number;
  leafValue?: number;
  x: number;
  y: number;
}

function ForestMapInner({ data }: Props) {
  const { nodes, edges, biggestTreeNodeIds } = useMemo(() => {
    const layout = layoutForest(data);

    // Find the tree with the highest influence
    let maxIdx = 0;
    let maxInf = -1;
    for (const inf of data.influence) {
      if (inf.mean_abs > maxInf) {
        maxInf = inf.mean_abs;
        maxIdx = inf.round;
      }
    }
    // The biggest tree index (for trees_per_round=1, round index = tree index)
    const biggestTreeIndex = maxIdx * data.trees_per_round;
    const prefix = `${biggestTreeIndex}-`;
    const ids = layout.nodes
      .filter((n) => n.id.startsWith(prefix))
      .map((n) => n.id);

    return { ...layout, biggestTreeNodeIds: ids };
  }, [data]);

  const [hover, setHover] = useState<HoverInfo | null>(null);
  const { fitView } = useReactFlow();

  // On initial render, zoom to biggest tree
  const onInit = useCallback(() => {
    if (biggestTreeNodeIds.length > 0) {
      setTimeout(() => {
        fitView({ nodes: biggestTreeNodeIds.map((id) => ({ id })), padding: 0.3, duration: 600 });
      }, 100);
    }
  }, [fitView, biggestTreeNodeIds]);

  const onNodeMouseEnter: NodeMouseHandler = useCallback(
    (event: React.MouseEvent, node: Node) => {
      const d = node.data as {
        nodeType?: string;
        treeIndex?: number;
        nodeId?: string;
        featureName?: string;
        threshold?: number;
        leafValue?: number;
      };
      if ((d.nodeType === "leaf" || d.nodeType === "split") && d.treeIndex !== undefined && d.nodeId) {
        setHover({
          treeIndex: d.treeIndex as number,
          nodeId: d.nodeId as string,
          nodeType: d.nodeType as "split" | "leaf",
          featureName: d.featureName,
          threshold: d.threshold,
          leafValue: d.leafValue,
          x: event.clientX,
          y: event.clientY,
        });
      }
    },
    [],
  );

  const onNodeMouseLeave = useCallback(() => setHover(null), []);

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        minZoom={0.02}
        maxZoom={4}
        onInit={onInit}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
      >
        <MiniMap
          nodeStrokeWidth={3}
          zoomable
          pannable
          style={{ background: "#f8f9fa" }}
        />
        <Background gap={50} size={1} color="#eee" />
      </ReactFlow>
      {hover && (
        <LeafTooltip
          treeIndex={hover.treeIndex}
          nodeId={hover.nodeId}
          nodeType={hover.nodeType}
          featureName={hover.featureName}
          threshold={hover.threshold}
          leafValue={hover.leafValue}
          x={hover.x}
          y={hover.y}
        />
      )}
    </div>
  );
}

export default function ForestMap({ data }: Props) {
  return (
    <ReactFlowProvider>
      <ForestMapInner data={data} />
    </ReactFlowProvider>
  );
}
