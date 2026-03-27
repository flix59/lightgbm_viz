import { useCallback, useMemo, useRef, useState } from "react";
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

import type { ForestResponse, TreeNode as TreeNodeType } from "../types/forest";
import type { SamplesData } from "../api/client";
import { layoutForest } from "../layout/forestLayout";
import SplitNode from "./SplitNode";
import LeafNode from "./LeafNode";
import LabelNode from "./LabelNode";
import LeafTooltip from "./LeafTooltip";

const nodeTypes = { splitNode: SplitNode, leafNode: LeafNode, labelNode: LabelNode };

interface Props {
  data: ForestResponse;
  samples: SamplesData;
}

interface PinInfo {
  treeIndex: number;
  nodeId: string;
  x: number;
  y: number;
}

function ForestMapInner({ data, samples }: Props) {
  const { nodes, edges, biggestTreeIndex } = useMemo(() => layoutForest(data), [data]);

  // Build lookup maps for tree data and node data
  const treeMap = useMemo(() => {
    const m = new Map<number, (typeof data.trees)[0]>();
    for (const t of data.trees) m.set(t.tree_index, t);
    return m;
  }, [data]);

  const treeNodeMap = useMemo(() => {
    const m = new Map<string, TreeNodeType>();
    for (const t of data.trees) {
      for (const n of t.nodes) {
        m.set(`${t.tree_index}-${n.id}`, n);
      }
    }
    return m;
  }, [data]);

  const [pinned, setPinned] = useState<PinInfo | null>(null);
  const [hover, setHover] = useState<PinInfo | null>(null);
  const { fitView } = useReactFlow();
  const initDone = useRef(false);

  const onInit = useCallback(() => {
    if (initDone.current) return;
    initDone.current = true;
    const prefix = `${biggestTreeIndex}-`;
    const treeNodeIds = nodes.filter((n) => n.id.startsWith(prefix)).map((n) => ({ id: n.id }));
    if (treeNodeIds.length > 0) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          fitView({ nodes: treeNodeIds, padding: 0.15, duration: 800 });
        });
      });
    }
  }, [fitView, biggestTreeIndex, nodes]);

  const extractInfo = useCallback((event: React.MouseEvent, node: Node): PinInfo | null => {
    const d = node.data as { nodeType?: string; treeIndex?: number; nodeId?: string };
    if ((d.nodeType === "leaf" || d.nodeType === "split") && d.treeIndex !== undefined && d.nodeId) {
      return { treeIndex: d.treeIndex as number, nodeId: d.nodeId as string, x: event.clientX, y: event.clientY };
    }
    return null;
  }, []);

  const onNodeMouseEnter: NodeMouseHandler = useCallback(
    (event: React.MouseEvent, node: Node) => {
      if (pinned) return;
      const info = extractInfo(event, node);
      if (info) setHover(info);
    },
    [pinned, extractInfo],
  );

  const onNodeMouseLeave = useCallback(() => { if (!pinned) setHover(null); }, [pinned]);

  const onNodeClick: NodeMouseHandler = useCallback(
    (event: React.MouseEvent, node: Node) => {
      const info = extractInfo(event, node);
      if (!info) return;
      if (pinned && pinned.treeIndex === info.treeIndex && pinned.nodeId === info.nodeId) {
        setPinned(null);
      } else {
        setPinned(info);
        setHover(null);
      }
    },
    [pinned, extractInfo],
  );

  const onPaneClick = useCallback(() => setPinned(null), []);

  const active = pinned || hover;
  const activeTree = active ? treeMap.get(active.treeIndex) : null;
  const activeNode = active ? treeNodeMap.get(`${active.treeIndex}-${active.nodeId}`) : null;

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
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
      >
        <MiniMap nodeStrokeWidth={3} zoomable pannable style={{ background: "#f8f9fa" }} />
        <Background gap={50} size={1} color="#eee" />
      </ReactFlow>
      {active && activeTree && activeNode && (
        <LeafTooltip
          key={`${active.treeIndex}-${active.nodeId}`}
          tree={activeTree}
          node={activeNode}
          samples={samples}
          x={active.x}
          y={active.y}
        />
      )}
    </div>
  );
}

export default function ForestMap({ data, samples }: Props) {
  return (
    <ReactFlowProvider>
      <ForestMapInner data={data} samples={samples} />
    </ReactFlowProvider>
  );
}
