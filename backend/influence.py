"""Per-boosting-round prediction influence computation.

Influence for boosting round t is defined as the incremental prediction delta:

    Δ_t(x) = F_t(x) - F_{t-1}(x)

where F_t(x) is the cumulative model output after t boosting rounds.

Because GBDT predictions are strictly additive sums of leaf values,
this equals the raw output of the K trees belonging to round t
(K = trees_per_round = 1 for regression/binary, n_classes for multiclass):

    Δ_t(x) = predict(x, start_iteration=t*K, num_iteration=K, raw_score=True)

We work in raw/margin space (raw_score=True for binary/multiclass) so that
the predictions remain additive and:

    init_score + Σ_t Δ_t(x) = F_T(x)   ← strict equality

For regression raw_score=True is a no-op (no link function).

Efficiency: one predict call per round, results cached by (split, sample_n).
Re-aggregating with a different metric is a cheap numpy operation on the cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def detect_objective(booster) -> str:
    """Return 'regression', 'binary', or 'multiclass'."""
    obj = booster.dump_model().get('objective', '')
    if isinstance(obj, list):
        obj = obj[0] if obj else ''
    s = obj.lower()
    if 'binary' in s or 'cross_entropy' in s or 'logloss' in s:
        return 'binary'
    if 'multiclass' in s or 'softmax' in s or 'multiclassova' in s:
        return 'multiclass'
    return 'regression'


def get_num_classes(booster) -> int:
    """Number of classes; 1 for regression and binary."""
    return max(1, int(booster.dump_model().get('num_class', 1)))


def get_trees_per_round(booster) -> int:
    """Trees added per boosting round; equals n_classes for multiclass, 1 otherwise."""
    return booster.num_model_per_iteration()


def get_num_rounds(booster) -> int:
    """Total boosting rounds = num_trees / trees_per_round."""
    return booster.num_trees() // get_trees_per_round(booster)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(
    booster,
    X: pd.DataFrame | np.ndarray,
    objective: str,
    trees_per_round: int,
    n_rounds: int,
) -> np.ndarray:
    """Compute per-round prediction deltas in raw/margin space.

    Indexing (0-based):
        delta[t] = contribution of boosting round t
                 = trees [t*K, t*K + K - 1] where K = trees_per_round

    Returns:
        regression / binary : (n_rounds, n_samples)
        multiclass          : (n_rounds, n_samples, n_classes)

    For regression raw_score has no effect (identity link).
    For binary/multiclass raw_score=True keeps predictions additive
    (working in log-odds / logit space rather than probability space).
    """
    use_raw = objective in ('binary', 'multiclass')
    n_classes = get_num_classes(booster) if objective == 'multiclass' else 1
    n_samples = X.shape[0]
    is_multiclass = (objective == 'multiclass' and n_classes > 1)

    out_shape = (n_rounds, n_samples, n_classes) if is_multiclass else (n_rounds, n_samples)
    deltas = np.empty(out_shape, dtype=np.float64)

    for t in range(n_rounds):
        raw = booster.predict(
            X,
            start_iteration=t * trees_per_round,
            num_iteration=trees_per_round,
            raw_score=use_raw,
        )
        deltas[t] = raw  # shape matches for both cases

    return deltas


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRICS = ('mean_abs', 'rms', 'signed_mean')

def aggregate_deltas(deltas: np.ndarray, metric: str) -> np.ndarray:
    """Reduce per-sample deltas to one scalar per boosting round.

    For multiclass (ndim=3) the class axis is averaged before aggregation,
    giving a single influence value per round regardless of objective.

    Args:
        deltas: (n_rounds, n_samples) or (n_rounds, n_samples, n_classes)
        metric: 'mean_abs' | 'rms' | 'signed_mean'

    Returns:
        shape (n_rounds,)
    """
    d = deltas.mean(axis=-1) if deltas.ndim == 3 else deltas

    if metric == 'mean_abs':
        return np.abs(d).mean(axis=1)
    if metric == 'rms':
        return np.sqrt((d ** 2).mean(axis=1))
    if metric == 'signed_mean':
        return d.mean(axis=1)
    raise ValueError(f"Unknown metric '{metric}'. Valid: {METRICS}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_influence_table(booster, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Compute all three metrics for every boosting round.

    Deltas are computed once; all metrics are derived from the same array.

    Columns:
        round         int    0-based boosting round index
        mean_abs      float  mean |Δ_t(x)| across samples
        rms           float  sqrt(mean Δ_t(x)²) across samples
        signed_mean   float  mean Δ_t(x) — can be negative (directional)
        pct_of_total  float  mean_abs as % of sum(mean_abs) across rounds
        cum_pct       float  cumulative % up to and including this round
    """
    obj = detect_objective(booster)
    k = get_trees_per_round(booster)
    T = get_num_rounds(booster)

    deltas = compute_deltas(booster, X, obj, k, T)

    ma = aggregate_deltas(deltas, 'mean_abs')
    rm = aggregate_deltas(deltas, 'rms')
    sm = aggregate_deltas(deltas, 'signed_mean')

    total = ma.sum()
    pct = ma / total * 100 if total > 0 else np.zeros(T)

    return pd.DataFrame({
        'round':        np.arange(T),
        'mean_abs':     ma,
        'rms':          rm,
        'signed_mean':  sm,
        'pct_of_total': pct,
        'cum_pct':      np.cumsum(pct),
    })


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def verify_additivity(
    booster,
    X: pd.DataFrame | np.ndarray,
    n_check: int = 5,
    rtol: float = 1e-5,
) -> dict:
    """Verify that Δ_t == F_{t+1}(x) − F_t(x) for a sample of rounds.

    Checks per-round rather than cumulative-sum-vs-final, which avoids
    ambiguity around the model's init score (LightGBM returns 0 rather
    than the base score when num_iteration=0, making a cumulative check
    misleading for regression on non-zero targets).

    Uses relative tolerance so the check is scale-independent.

    Args:
        n_check: number of rounds to spot-check (spread evenly).
        rtol:    relative tolerance — passes if max_rel_err < rtol.

    Returns dict with 'max_rel_err', 'mean_rel_err', 'passed'.
    """
    obj     = detect_objective(booster)
    k       = get_trees_per_round(booster)
    T       = get_num_rounds(booster)
    use_raw = obj in ('binary', 'multiclass')

    rounds = np.linspace(0, T - 1, min(n_check, T), dtype=int)
    rel_errs: list[float] = []

    for t in rounds:
        # Our delta for round t
        delta_t = booster.predict(
            X, start_iteration=t * k, num_iteration=k, raw_score=use_raw,
        )
        # Cumulative difference: F_{t+1} - F_t
        F_next = booster.predict(X, num_iteration=(t + 1) * k, raw_score=use_raw)
        F_curr = booster.predict(X, num_iteration=t * k,       raw_score=use_raw)
        diff   = F_next - F_curr

        scale = np.abs(diff).mean() + 1e-12   # avoid div-by-zero on flat rounds
        rel_errs.append(float(np.abs(delta_t - diff).mean() / scale))

    max_rel  = float(np.max(rel_errs))
    mean_rel = float(np.mean(rel_errs))
    return {
        'max_rel_err':  max_rel,
        'mean_rel_err': mean_rel,
        'passed':       max_rel < rtol,
    }
