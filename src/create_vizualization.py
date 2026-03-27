"""Precompute all data and build a single self-contained HTML file.

Usage:
    uv run python create_vizualization.py
    uv run python create_vizualization.py --model path/to/model.pkl --data path/to/train_data.pkl
    uv run python create_vizualization.py --output my_viz.html
"""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from tree_extractor import TreeExtractor
from influence import (
    detect_objective,
    get_trees_per_round,
    get_num_rounds,
    compute_deltas,
    aggregate_deltas,
)

ROOT = Path(__file__).parent
FRONTEND = ROOT.parent / "frontend"

SHAP_MAX_SAMPLES = 500
PDP_GRID_SIZE = 50
PDP_MAX_SAMPLES = 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    model: str = "model.pkl",
    data: str = "train_data.pkl",
    output: str = "forest_viz.html",
    shap_samples: int = SHAP_MAX_SAMPLES,
    pdp_samples: int = PDP_MAX_SAMPLES,
    pdp_grid: int = PDP_GRID_SIZE,
):
    """Generate a self-contained HTML visualization of a LightGBM model.

    Args:
        model: Path to the pickled LightGBM booster.
        data: Path to the pickled training data (X_train, y_train, X_test, y_test).
        output: Output HTML file path.
        shap_samples: Max samples for SHAP computation.
        pdp_samples: Max samples for PDP computation.
        pdp_grid: Number of grid points for PDP curves.
    """
    booster, X_train, y_train = load_model_and_data(model, data)
    extractor = TreeExtractor(booster, X_train, y_train)
    feature_names = extractor.feature_names

    forest = compute_forest(extractor, booster, X_train, y_train)
    shap_data = compute_shap(booster, X_train, feature_names, shap_samples)
    pdp_data = compute_pdp(booster, X_train, feature_names, pdp_samples, pdp_grid)
    samples = export_samples(X_train, y_train, feature_names)

    write_dev_json(FRONTEND / "public" / "data", forest, shap_data, pdp_data, samples)
    build_frontend()
    assemble_html(forest, shap_data, pdp_data, samples, output)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_and_data(model_path: str, data_path: str):
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        booster = pickle.load(f)
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        X_train, y_train, _X_test, _y_test = pickle.load(f)
    return booster, X_train, y_train


# ---------------------------------------------------------------------------
# Forest (trees + influence)
# ---------------------------------------------------------------------------


def compute_forest(
    extractor: TreeExtractor, booster, X_train: pd.DataFrame, y_train
) -> dict:
    num_trees = extractor.get_num_trees()
    objective = detect_objective(booster)
    trees_per_round = get_trees_per_round(booster)
    n_rounds = get_num_rounds(booster)
    feature_names = extractor.feature_names

    print(f"Objective: {objective} | Trees: {num_trees} | Rounds: {n_rounds}")

    influence = compute_influence(
        booster, X_train, objective, trees_per_round, n_rounds
    )

    print("Extracting tree structures...")
    trees = [flatten_tree(i, extractor, trees_per_round) for i in range(num_trees)]

    return {
        "num_trees": num_trees,
        "num_rounds": n_rounds,
        "trees_per_round": trees_per_round,
        "objective": objective,
        "feature_names": feature_names,
        "influence": influence,
        "trees": trees,
    }


def compute_influence(
    booster, X_train: pd.DataFrame, objective: str, trees_per_round: int, n_rounds: int
) -> list[dict]:
    print("Computing influence deltas...")
    deltas = compute_deltas(booster, X_train, objective, trees_per_round, n_rounds)
    ma = aggregate_deltas(deltas, "mean_abs")
    rm = aggregate_deltas(deltas, "rms")
    sm = aggregate_deltas(deltas, "signed_mean")
    total = ma.sum()
    pct = ma / total * 100 if total > 0 else np.zeros(n_rounds)
    cum_pct = np.cumsum(pct)
    return [
        {
            "round": i,
            "mean_abs": float(ma[i]),
            "rms": float(rm[i]),
            "signed_mean": float(sm[i]),
            "pct_of_total": float(pct[i]),
            "cum_pct": float(cum_pct[i]),
        }
        for i in range(n_rounds)
    ]


def flatten_tree(tree_idx: int, extractor: TreeExtractor, trees_per_round: int) -> dict:
    tree = extractor.get_tree(tree_idx)
    node_samples = extractor.calculate_node_samples(tree_idx)
    positions = calc_pos(tree)
    nodes: list[dict] = []

    def visit(node: dict):
        if "split_index" not in node:
            nid = f"leaf_{node.get('leaf_index', 'unknown')}"
            rx, ry = positions[nid]
            nodes.append(
                {
                    "id": nid,
                    "type": "leaf",
                    "rel_x": rx,
                    "rel_y": ry,
                    "sample_count": node_samples.get(nid, 0),
                    "leaf_value": node.get("leaf_value", 0.0),
                    "children": [],
                }
            )
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
            nodes.append(
                {
                    "id": nid,
                    "type": "split",
                    "rel_x": rx,
                    "rel_y": ry,
                    "sample_count": node_samples.get(nid, 0),
                    "feature_name": extractor.feature_names[fidx],
                    "feature_index": fidx,
                    "threshold": node["threshold"],
                    "split_gain": node.get("split_gain", 0.0),
                    "children": children,
                }
            )
            visit(node["left_child"])
            visit(node["right_child"])

    visit(tree)
    return {
        "tree_index": tree_idx,
        "round_index": tree_idx // trees_per_round,
        "nodes": nodes,
    }


def calc_pos(node: dict) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}

    def walk(n, depth=0, pos=0) -> int:
        if "split_index" not in n:
            nid = f"leaf_{n.get('leaf_index', 'unknown')}"
            positions[nid] = (pos, -depth)
            return 2
        nid = f"split_{n['split_index']}"
        lw = walk(n["left_child"], depth + 1, pos)
        rw = walk(n["right_child"], depth + 1, pos + lw)
        lc, rc = n["left_child"], n["right_child"]
        lk = (
            f"split_{lc['split_index']}"
            if "split_index" in lc
            else f"leaf_{lc.get('leaf_index', 'unknown')}"
        )
        rk = (
            f"split_{rc['split_index']}"
            if "split_index" in rc
            else f"leaf_{rc.get('leaf_index', 'unknown')}"
        )
        positions[nid] = ((positions[lk][0] + positions[rk][0]) / 2, -depth)
        return lw + rw

    walk(node)
    return positions


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def compute_shap(
    booster, X_train: pd.DataFrame, feature_names: list[str], shap_samples: int
) -> dict:
    print("Computing SHAP values...")
    X_shap = _subsample(X_train, shap_samples)
    contribs = booster.predict(X_shap, pred_contrib=True)
    base_value = float(contribs[0, -1])
    shap_matrix = contribs[:, :-1]
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(-mean_abs_shap)
    return {
        "base_value": base_value,
        "features": [
            {
                "feature_name": feature_names[fi],
                "mean_abs_shap": float(mean_abs_shap[fi]),
                "values": X_shap.iloc[:, fi].values.astype(float).tolist(),
                "shap_values": shap_matrix[:, fi].astype(float).tolist(),
            }
            for fi in order
        ],
    }


# ---------------------------------------------------------------------------
# PDP
# ---------------------------------------------------------------------------


def compute_pdp(
    booster,
    X_train: pd.DataFrame,
    feature_names: list[str],
    pdp_samples: int,
    pdp_grid: int,
) -> dict:
    print("Computing partial dependence plots...")
    X_pdp = _subsample(X_train, pdp_samples).copy()
    curves = []
    for fname in feature_names:
        col = X_pdp[fname].values.astype(float)
        grid = np.linspace(float(col.min()), float(col.max()), pdp_grid)
        avg_preds = []
        for g in grid:
            X_mod = X_pdp.copy()
            X_mod[fname] = g
            avg_preds.append(float(booster.predict(X_mod).mean()))
        curves.append(
            {
                "feature_name": fname,
                "grid": grid.tolist(),
                "avg_prediction": avg_preds,
                "feature_values": col.tolist(),
            }
        )
    return {"curves": curves}


def _subsample(X: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        return X.iloc[idx]
    return X


# ---------------------------------------------------------------------------
# Samples export
# ---------------------------------------------------------------------------


def export_samples(X_train: pd.DataFrame, y_train, feature_names: list[str]) -> dict:
    print("Exporting training samples...")
    return {
        "features": {
            fname: X_train[fname].values.astype(float).round(4).tolist()
            for fname in feature_names
        },
        "target": y_train.values.astype(float).round(2).tolist(),
    }


# ---------------------------------------------------------------------------
# Frontend build & HTML assembly
# ---------------------------------------------------------------------------


def write_dev_json(
    data_dir: Path, forest: dict, shap_data: dict, pdp_data: dict, samples: dict
):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("forest.json", forest),
        ("shap.json", shap_data),
        ("pdp.json", pdp_data),
        ("samples.json", samples),
    ]:
        (data_dir / name).write_text(json.dumps(obj, separators=(",", ":")))


def build_frontend():
    print("Building frontend...")
    subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND,
        check=True,
        capture_output=True,
        text=True,
    )


def assemble_html(
    forest: dict, shap_data: dict, pdp_data: dict, samples: dict, output: str
):
    print(f"Assembling {output}...")
    dist = FRONTEND / "dist"

    css_files = list((dist / "assets").glob("*.css"))
    js_files = list((dist / "assets").glob("*.js"))

    css_content = "\n".join(f.read_text() for f in css_files)
    js_content = "\n".join(f.read_text() for f in js_files)

    j = json.dumps
    forest_json = j(forest, separators=(",", ":"))
    shap_json = j(shap_data, separators=(",", ":"))
    pdp_json = j(pdp_data, separators=(",", ":"))
    samples_json = j(samples, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LightGBM Forest Visualizer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body, #root {{ width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
{css_content}
</style>
</head>
<body>
<div id="root"></div>
<script>
window.__FOREST__ = {forest_json};
window.__SHAP__ = {shap_json};
window.__PDP__ = {pdp_json};
window.__SAMPLES__ = {samples_json};
</script>
<script>{js_content}</script>
</body>
</html>"""

    out_path = Path(output)
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  {output}: {size_mb:.1f} MB")
    print("  Open it directly in your browser — no server needed!")


if __name__ == "__main__":
    fire.Fire(main)
