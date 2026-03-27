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

// SHAP
export interface ShapFeatureSummary {
  feature_name: string;
  mean_abs_shap: number;
  values: number[];
  shap_values: number[];
}

export interface ShapSummaryResponse {
  features: ShapFeatureSummary[];
  base_value: number;
}

// PDP
export interface PdpCurve {
  feature_name: string;
  grid: number[];
  avg_prediction: number[];
  feature_values: number[];
}

export interface PdpResponse {
  curves: PdpCurve[];
}
