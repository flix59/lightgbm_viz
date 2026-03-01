import type { ForestResponse, NodeDistributionResponse } from "../types/forest";

export async function fetchForest(): Promise<ForestResponse> {
  const res = await fetch("/api/forest");
  if (!res.ok) throw new Error(`Failed to fetch forest: ${res.status}`);
  return res.json();
}

const distCache = new Map<string, NodeDistributionResponse>();

export async function fetchNodeDistribution(
  treeIndex: number,
  nodeId: string,
): Promise<NodeDistributionResponse> {
  const key = `${treeIndex}/${nodeId}`;
  const cached = distCache.get(key);
  if (cached) return cached;

  const res = await fetch(`/api/node/${treeIndex}/${nodeId}/distribution`);
  if (!res.ok) throw new Error(`Failed to fetch node distribution: ${res.status}`);
  const data: NodeDistributionResponse = await res.json();
  distCache.set(key, data);
  return data;
}
