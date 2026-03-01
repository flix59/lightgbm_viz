export interface TreeNode {
  id: string;
  type: "split" | "leaf";
  rel_x: number;
  rel_y: number;
  sample_count: number;
  feature_name?: string;
  feature_index?: number;
  threshold?: number;
  split_gain?: number;
  leaf_value?: number;
  children: string[];
}

export interface TreeData {
  tree_index: number;
  round_index: number;
  nodes: TreeNode[];
}

export interface RoundInfluence {
  round: number;
  mean_abs: number;
  rms: number;
  signed_mean: number;
  pct_of_total: number;
  cum_pct: number;
}

export interface ForestResponse {
  num_trees: number;
  num_rounds: number;
  trees_per_round: number;
  objective: string;
  feature_names: string[];
  influence: RoundInfluence[];
  trees: TreeData[];
}

export interface ScatterData {
  left_x: number[];
  left_y: number[];
  right_x: number[];
  right_y: number[];
  feature_name: string;
  threshold: number;
}

export interface NodeDistributionResponse {
  histogram: {
    bin_edges: number[];
    counts: number[];
  };
  target_stats: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    q25: number;
    q75: number;
  };
  scatter: ScatterData | null;
}
