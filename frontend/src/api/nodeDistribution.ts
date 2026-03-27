import type { TreeData, TreeNode } from "../types/forest";
import type { SamplesData } from "./client";

export interface NodeDistribution {
  // Always present
  sampleCount: number;
  targetMean: number;
  targetMedian: number;
  targetStd: number;
  targetMin: number;
  targetMax: number;
  histCounts: number[];
  histEdges: number[];
  // Only for split nodes
  scatter?: {
    featureName: string;
    threshold: number;
    leftX: number[];
    leftY: number[];
    rightX: number[];
    rightY: number[];
  };
}

const MAX_SCATTER = 500;

/**
 * Traverse a tree from root to find sample indices reaching a given node.
 */
function getSampleIndices(
  tree: TreeData,
  targetNodeId: string,
  samples: SamplesData,
): number[] | null {
  const nodeMap = new Map<string, TreeNode>();
  for (const n of tree.nodes) nodeMap.set(n.id, n);

  // BFS/DFS from root (first node) tracking sample indices
  const root = tree.nodes[0];
  if (!root) return null;

  function traverse(nodeId: string, indices: number[]): number[] | null {
    if (nodeId === targetNodeId) return indices;

    const node = nodeMap.get(nodeId);
    if (!node || node.type !== "split" || node.children.length < 2) return null;

    const featureVals = samples.features[node.feature_name!];
    const threshold = node.threshold!;

    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (const i of indices) {
      if (featureVals[i] <= threshold) leftIdx.push(i);
      else rightIdx.push(i);
    }

    // Try left child, then right child
    const leftResult = traverse(node.children[0], leftIdx);
    if (leftResult) return leftResult;
    return traverse(node.children[1], rightIdx);
  }

  const allIndices = Array.from({ length: samples.target.length }, (_, i) => i);
  return traverse(root.id, allIndices);
}

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function histogram(values: number[], bins: number = 20): { counts: number[]; edges: number[] } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = range / bins;
  const edges: number[] = [];
  for (let i = 0; i <= bins; i++) edges.push(min + i * step);
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let bin = Math.floor((v - min) / step);
    if (bin >= bins) bin = bins - 1;
    counts[bin]++;
  }
  return { counts, edges };
}

const cache = new Map<string, NodeDistribution>();

export function computeNodeDistribution(
  tree: TreeData,
  nodeId: string,
  node: TreeNode,
  samples: SamplesData,
): NodeDistribution | null {
  const key = `${tree.tree_index}-${nodeId}`;
  const cached = cache.get(key);
  if (cached) return cached;

  const indices = getSampleIndices(tree, nodeId, samples);
  if (!indices || indices.length === 0) return null;

  const targetVals = indices.map((i) => samples.target[i]);
  const n = targetVals.length;
  const mean = targetVals.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(targetVals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const sorted = [...targetVals].sort((a, b) => a - b);
  const hist = histogram(targetVals);

  const result: NodeDistribution = {
    sampleCount: n,
    targetMean: mean,
    targetMedian: median(targetVals),
    targetStd: std,
    targetMin: sorted[0],
    targetMax: sorted[n - 1],
    histCounts: hist.counts,
    histEdges: hist.edges,
  };

  // Scatter for split nodes
  if (node.type === "split" && node.feature_name && node.threshold != null) {
    const featureVals = indices.map((i) => samples.features[node.feature_name!][i]);
    const threshold = node.threshold;

    // Downsample if needed
    let useIndices: number[];
    if (indices.length > MAX_SCATTER) {
      // Simple deterministic downsample
      const step = indices.length / MAX_SCATTER;
      useIndices = Array.from({ length: MAX_SCATTER }, (_, i) => Math.floor(i * step));
    } else {
      useIndices = Array.from({ length: indices.length }, (_, i) => i);
    }

    const leftX: number[] = [], leftY: number[] = [];
    const rightX: number[] = [], rightY: number[] = [];
    for (const i of useIndices) {
      const fv = featureVals[i];
      const tv = targetVals[i];
      if (fv <= threshold) { leftX.push(fv); leftY.push(tv); }
      else { rightX.push(fv); rightY.push(tv); }
    }

    result.scatter = { featureName: node.feature_name, threshold, leftX, leftY, rightX, rightY };
  }

  cache.set(key, result);
  return result;
}
