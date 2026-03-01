"""FastAPI backend for LightGBM forest visualization."""
from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ForestResponse,
    HistogramData,
    NodeDistributionResponse,
    RoundInfluence,
    ScatterData,
    TargetStats,
    TreeData,
    TreeNode,
)
from .tree_extractor import TreeExtractor
from .influence import (
    detect_objective,
    get_trees_per_round,
    get_num_rounds,
    compute_deltas,
    aggregate_deltas,
)

# ---------------------------------------------------------------------------
# Global state populated at startup
# ---------------------------------------------------------------------------
_state: dict = {}

ROOT = Path(__file__).resolve().parent.parent


def _calc_pos(node: dict) -> tuple[dict[str, tuple[float, float]], int]:
    """Compute relative (x, y) for every node in a tree. Returns (positions, width)."""
    positions: dict[str, tuple[float, float]] = {}

    def _walk(n, depth=0, pos=0) -> int:
        if "split_index" not in n:
            nid = f"leaf_{n.get('leaf_index', 'unknown')}"
            positions[nid] = (pos, -depth)
            return 2
        nid = f"split_{n['split_index']}"
        lw = _walk(n["left_child"], depth + 1, pos)
        rw = _walk(n["right_child"], depth + 1, pos + lw)
        lc, rc = n["left_child"], n["right_child"]
        lk = f"split_{lc['split_index']}" if "split_index" in lc else f"leaf_{lc.get('leaf_index', 'unknown')}"
        rk = f"split_{rc['split_index']}" if "split_index" in rc else f"leaf_{rc.get('leaf_index', 'unknown')}"
        positions[nid] = ((positions[lk][0] + positions[rk][0]) / 2, -depth)
        return lw + rw

    w = _walk(node)
    return positions, w


def _flatten_tree(
    tree_idx: int,
    extractor: TreeExtractor,
    trees_per_round: int,
) -> TreeData:
    """Convert a LightGBM tree dict into a flat list of TreeNode."""
    tree = extractor.get_tree(tree_idx)
    node_samples = extractor.calculate_node_samples(tree_idx)
    positions, _ = _calc_pos(tree)

    nodes: list[TreeNode] = []

    def _visit(node: dict):
        if "split_index" not in node:
            nid = f"leaf_{node.get('leaf_index', 'unknown')}"
            rx, ry = positions[nid]
            nodes.append(TreeNode(
                id=nid,
                type="leaf",
                rel_x=rx,
                rel_y=ry,
                sample_count=node_samples.get(nid, 0),
                leaf_value=node.get("leaf_value", 0.0),
            ))
        else:
            nid = f"split_{node['split_index']}"
            rx, ry = positions[nid]
            fidx = node["split_feature"]
            children = []
            for ck in ("left_child", "right_child"):
                child = node[ck]
                cid = (
                    f"split_{child['split_index']}"
                    if "split_index" in child
                    else f"leaf_{child.get('leaf_index', 'unknown')}"
                )
                children.append(cid)
            nodes.append(TreeNode(
                id=nid,
                type="split",
                rel_x=rx,
                rel_y=ry,
                sample_count=node_samples.get(nid, 0),
                feature_name=extractor.feature_names[fidx],
                feature_index=fidx,
                threshold=node["threshold"],
                split_gain=node.get("split_gain", 0.0),
                children=children,
            ))
            _visit(node["left_child"])
            _visit(node["right_child"])

    _visit(tree)
    return TreeData(
        tree_index=tree_idx,
        round_index=tree_idx // trees_per_round,
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and data...")
    with open(ROOT / "model.pkl", "rb") as f:
        booster = pickle.load(f)
    with open(ROOT / "train_data.pkl", "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    extractor = TreeExtractor(booster, X_train, y_train)
    num_trees = extractor.get_num_trees()
    objective = detect_objective(booster)
    trees_per_round = get_trees_per_round(booster)
    n_rounds = get_num_rounds(booster)

    print(f"Objective: {objective} | Trees: {num_trees} | Rounds: {n_rounds}")

    # Pre-compute influence
    print("Computing influence deltas...")
    deltas = compute_deltas(booster, X_train, objective, trees_per_round, n_rounds)
    ma = aggregate_deltas(deltas, "mean_abs")
    rm = aggregate_deltas(deltas, "rms")
    sm = aggregate_deltas(deltas, "signed_mean")
    total = ma.sum()
    pct = ma / total * 100 if total > 0 else np.zeros(n_rounds)

    influence = [
        RoundInfluence(
            round=i,
            mean_abs=float(ma[i]),
            rms=float(rm[i]),
            signed_mean=float(sm[i]),
            pct_of_total=float(pct[i]),
            cum_pct=float(np.cumsum(pct)[i]),
        )
        for i in range(n_rounds)
    ]

    # Pre-compute all trees
    print("Extracting tree structures...")
    trees = [_flatten_tree(i, extractor, trees_per_round) for i in range(num_trees)]

    _state["booster"] = booster
    _state["extractor"] = extractor
    _state["y_train"] = y_train
    _state["num_trees"] = num_trees
    _state["n_rounds"] = n_rounds
    _state["trees_per_round"] = trees_per_round
    _state["objective"] = objective
    _state["feature_names"] = extractor.feature_names
    _state["influence"] = influence
    _state["trees"] = trees

    # Build the full forest response once
    _state["forest_response"] = ForestResponse(
        num_trees=num_trees,
        num_rounds=n_rounds,
        trees_per_round=trees_per_round,
        objective=objective,
        feature_names=extractor.feature_names,
        influence=influence,
        trees=trees,
    )

    print("Ready.")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LightGBM Forest Viz", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/forest", response_model=ForestResponse)
def get_forest():
    return _state["forest_response"]


MAX_SCATTER_POINTS = 500


def _find_split_info(extractor: TreeExtractor, tree_index: int, node_id: str):
    """Walk the tree to find the feature_name and threshold for a split node."""
    tree = extractor.get_tree(tree_index)

    def _search(node):
        if "split_index" not in node:
            return None
        nid = f"split_{node['split_index']}"
        if nid == node_id:
            fidx = node["split_feature"]
            return extractor.feature_names[fidx], node["threshold"]
        for ck in ("left_child", "right_child"):
            result = _search(node[ck])
            if result:
                return result
        return None

    return _search(tree)


@app.get("/api/node/{tree_index}/{node_id}/distribution", response_model=NodeDistributionResponse)
def get_node_distribution(tree_index: int, node_id: str):
    extractor: TreeExtractor = _state["extractor"]
    y_data = _state["y_train"]

    if tree_index < 0 or tree_index >= _state["num_trees"]:
        raise HTTPException(status_code=404, detail="Tree index out of range")

    node_indices = extractor.get_node_sample_indices(tree_index)
    if node_id not in node_indices:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in tree {tree_index}")

    indices = node_indices[node_id]
    y_vals = y_data.iloc[indices].values.astype(float)

    counts, bin_edges = np.histogram(y_vals, bins=20)

    # Build scatter data for split nodes
    scatter = None
    if node_id.startswith("split_"):
        info = _find_split_info(extractor, tree_index, node_id)
        if info:
            feature_name, threshold = info
            x_vals = extractor.X_data[feature_name].iloc[indices].values.astype(float)

            # Downsample if too many points
            if len(x_vals) > MAX_SCATTER_POINTS:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(x_vals), MAX_SCATTER_POINTS, replace=False)
                x_vals = x_vals[sample_idx]
                y_scatter = y_vals[sample_idx]
            else:
                y_scatter = y_vals

            left_mask = x_vals <= threshold
            scatter = ScatterData(
                left_x=x_vals[left_mask].tolist(),
                left_y=y_scatter[left_mask].tolist(),
                right_x=x_vals[~left_mask].tolist(),
                right_y=y_scatter[~left_mask].tolist(),
                feature_name=feature_name,
                threshold=float(threshold),
            )

    return NodeDistributionResponse(
        histogram=HistogramData(
            bin_edges=bin_edges.tolist(),
            counts=counts.tolist(),
        ),
        target_stats=TargetStats(
            mean=float(np.mean(y_vals)),
            median=float(np.median(y_vals)),
            std=float(np.std(y_vals)),
            min=float(np.min(y_vals)),
            max=float(np.max(y_vals)),
            q25=float(np.percentile(y_vals, 25)),
            q75=float(np.percentile(y_vals, 75)),
        ),
        scatter=scatter,
    )
