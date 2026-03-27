import type { Node, Edge } from "@xyflow/react";
import type { ForestResponse, TreeData } from "../types/forest";

const TREE_GAP_X = 120;
const TREE_GAP_Y = 100;
const NODE_SPACING_X = 55;
const NODE_SPACING_Y = 80;
const MAX_TREES_PER_ROW = 10;
const MAX_EDGE_WIDTH = 12;
const MIN_EDGE_WIDTH = 0.5;

export interface LayoutResult {
  nodes: Node[];
  edges: Edge[];
  biggestTreeIndex: number;
}

export function layoutForest(data: ForestResponse): LayoutResult {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Find tree with highest influence
  let biggestTreeIndex = 0;
  let maxInf = -1;
  for (const inf of data.influence) {
    if (inf.mean_abs > maxInf) {
      maxInf = inf.mean_abs;
      biggestTreeIndex = inf.round * data.trees_per_round;
    }
  }

  // Compute influence-based scale per round
  const meanAbsValues = data.influence.map((inf) => inf.mean_abs);
  const lo = Math.min(...meanAbsValues);
  const hi = Math.max(...meanAbsValues);
  const range = hi - lo || 1;

  let globalOffsetX = 0;
  let globalOffsetY = 0;
  let rowMaxHeight = 0;
  let treesInRow = 0;

  for (const tree of data.trees) {
    const roundInf = data.influence[tree.round_index];
    const normalized = (roundInf.mean_abs - lo) / range;
    const scale = 0.35 + 0.65 * normalized;

    const sx = NODE_SPACING_X * scale;
    const sy = NODE_SPACING_Y * scale * -1; // negate: backend rel_y is -depth, we want +y = down

    // Find tree dimensions from node positions
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of tree.nodes) {
      const nx = n.rel_x * sx;
      const ny = n.rel_y * sy;
      if (nx < minX) minX = nx;
      if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny;
      if (ny > maxY) maxY = ny;
    }
    const treeWidth = maxX - minX + 100;
    const treeHeight = maxY - minY + 100;

    if (treesInRow >= MAX_TREES_PER_ROW) {
      globalOffsetX = 0;
      globalOffsetY += rowMaxHeight + TREE_GAP_Y;
      rowMaxHeight = 0;
      treesInRow = 0;
    }

    layoutTree(tree, globalOffsetX - minX, globalOffsetY - minY, sx, sy, scale, roundInf.pct_of_total, nodes, edges);

    globalOffsetX += treeWidth + TREE_GAP_X;
    if (treeHeight > rowMaxHeight) rowMaxHeight = treeHeight;
    treesInRow++;
  }

  return { nodes, edges, biggestTreeIndex };
}

function layoutTree(
  tree: TreeData,
  offsetX: number,
  offsetY: number,
  sx: number,
  sy: number,
  scale: number,
  pctOfTotal: number,
  nodes: Node[],
  edges: Edge[],
) {
  // Build a map from node id → sample_count for edge width lookup
  const sampleMap = new Map<string, number>();
  for (const n of tree.nodes) {
    sampleMap.set(n.id, n.sample_count);
  }
  const totalSamples = tree.nodes[0]?.sample_count || 1;

  // Add tree label node (custom type, no handles)
  const rootNode = tree.nodes[0];
  if (rootNode) {
    nodes.push({
      id: `tree-label-${tree.tree_index}`,
      type: "labelNode",
      position: { x: offsetX + rootNode.rel_x * sx - 30, y: offsetY + rootNode.rel_y * sy - 30 },
      data: {
        label: `T${tree.tree_index} (${pctOfTotal.toFixed(1)}%)`,
        fontSize: Math.max(8, 11 * scale),
        color: "#444",
      },
      selectable: false,
      draggable: false,
      connectable: false,
    });
  }

  const maxSamples = Math.max(...tree.nodes.map((n) => n.sample_count), 1);

  for (const n of tree.nodes) {
    const x = offsetX + n.rel_x * sx;
    const y = offsetY + n.rel_y * sy;
    const nodeId = `${tree.tree_index}-${n.id}`;

    const sizeScale = 0.4 + 0.6 * (n.sample_count / maxSamples);
    const baseSize = Math.max(30, 60 * scale);
    const size = baseSize * sizeScale;

    nodes.push({
      id: nodeId,
      type: n.type === "split" ? "splitNode" : "leafNode",
      position: { x: x - size / 2, y },
      data: {
        treeIndex: tree.tree_index,
        nodeId: n.id,
        nodeType: n.type,
        sampleCount: n.sample_count,
        featureName: n.feature_name,
        threshold: n.threshold,
        splitGain: n.split_gain,
        leafValue: n.leaf_value,
        width: size,
        height: size * 0.6,
        scale,
      },
      draggable: false,
      connectable: false,
    });

    // Edges to children — width proportional to child sample count
    for (const childId of n.children) {
      const edgeId = `${tree.tree_index}-${n.id}->${childId}`;
      const targetNodeId = `${tree.tree_index}-${childId}`;
      const childSamples = sampleMap.get(childId) || 0;
      const widthRatio = childSamples / totalSamples;
      const edgeWidth = Math.max(MIN_EDGE_WIDTH, widthRatio * MAX_EDGE_WIDTH * scale);

      edges.push({
        id: edgeId,
        source: nodeId,
        target: targetNodeId,
        type: "default",
        style: { stroke: "#888", strokeWidth: edgeWidth },
      });
    }
  }
}
