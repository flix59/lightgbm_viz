import type { ForestResponse, ShapSummaryResponse, PdpResponse } from "../types/forest";

export interface SamplesData {
  features: Record<string, number[]>;
  target: number[];
}

// In the single-HTML build, data is inlined as window globals.
// In dev mode, fall back to fetch.
declare global {
  interface Window {
    __FOREST__?: ForestResponse;
    __SHAP__?: ShapSummaryResponse;
    __PDP__?: PdpResponse;
    __SAMPLES__?: SamplesData;
  }
}

async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

export async function fetchForest(): Promise<ForestResponse> {
  return window.__FOREST__ ?? await loadJson("/data/forest.json");
}

export async function fetchShapSummary(): Promise<ShapSummaryResponse> {
  return window.__SHAP__ ?? await loadJson("/data/shap.json");
}

export async function fetchPdp(): Promise<PdpResponse> {
  return window.__PDP__ ?? await loadJson("/data/pdp.json");
}

export async function fetchSamples(): Promise<SamplesData> {
  return window.__SAMPLES__ ?? await loadJson("/data/samples.json");
}
