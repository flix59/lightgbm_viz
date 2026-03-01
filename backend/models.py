"""Pydantic response schemas for the forest API."""
from __future__ import annotations

from pydantic import BaseModel


class TreeNode(BaseModel):
    id: str
    type: str  # "split" or "leaf"
    rel_x: float
    rel_y: float
    sample_count: int
    # Split-only fields
    feature_name: str | None = None
    feature_index: int | None = None
    threshold: float | None = None
    split_gain: float | None = None
    # Leaf-only fields
    leaf_value: float | None = None
    # Children (node ids)
    children: list[str] = []


class TreeData(BaseModel):
    tree_index: int
    round_index: int
    nodes: list[TreeNode]


class RoundInfluence(BaseModel):
    round: int
    mean_abs: float
    rms: float
    signed_mean: float
    pct_of_total: float
    cum_pct: float


class ForestResponse(BaseModel):
    num_trees: int
    num_rounds: int
    trees_per_round: int
    objective: str
    feature_names: list[str]
    influence: list[RoundInfluence]
    trees: list[TreeData]


class HistogramData(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class TargetStats(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float


class ScatterData(BaseModel):
    left_x: list[float]
    left_y: list[float]
    right_x: list[float]
    right_y: list[float]
    feature_name: str
    threshold: float


class NodeDistributionResponse(BaseModel):
    histogram: HistogramData
    target_stats: TargetStats
    scatter: ScatterData | None = None
