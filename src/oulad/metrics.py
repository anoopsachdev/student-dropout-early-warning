"""Ranking metrics and the within-module outreach rule.

Recall at 20% is computed inside each module: a tutor can contact the top fifth
of that module, not the top fifth of the whole university. ROC-AUC and PR-AUC
stay global for a single model. Leave-one-module-out reports a weighted mean of
per-module scores, because scores from different models are not on one scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from oulad.config import BUDGET, MIN_GROUP_N, MIN_GROUP_POS


def flags_at_budget(scores, budget: float = BUDGET) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    flag = np.zeros(n, dtype=bool)
    k = int(np.floor(n * budget))
    if k <= 0 or n == 0:
        return flag
    order = np.argsort(-scores, kind="mergesort")
    flag[order[:k]] = True
    return flag


def within_module_flags(modules, scores, budget: float = BUDGET) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    modules = np.asarray(modules)
    flag = np.zeros(len(scores), dtype=bool)
    for module in pd.unique(modules):
        index = np.flatnonzero(modules == module)
        flag[index] = flags_at_budget(scores[index], budget)
    return flag


def recall_from_flags(y, flag) -> float | None:
    y = np.asarray(y).astype(bool)
    flag = np.asarray(flag).astype(bool)
    positives = int(y.sum())
    if positives == 0:
        return None
    return float((flag & y).sum() / positives)


def ranking_metrics(y, scores) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def rates(y, flag) -> dict:
    y = np.asarray(y).astype(bool)
    flag = np.asarray(flag).astype(bool)
    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    true_positive = int((flag & y).sum())
    false_negative = int((~flag & y).sum())
    false_positive = int((flag & ~y).sum())
    return {
        "n": n,
        "n_pos": n_pos,
        "base_rate": (n_pos / n) if n else None,
        "selection_rate": float(flag.mean()) if n else None,
        "fnr": (false_negative / n_pos) if n_pos else None,
        "fpr": (false_positive / n_neg) if n_neg else None,
        "recall": (true_positive / n_pos) if n_pos else None,
    }


def group_table(y, flag, groups) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    groups = np.asarray(groups)
    rows = []
    for group in pd.unique(groups):
        mask = groups == group
        rows.append({"group": group, **rates(y[mask], flag[mask])})
    return pd.DataFrame(rows)


def fnr_gap(
    table: pd.DataFrame,
    levels: list[str] | tuple[str, ...] | None = None,
    min_n: int = MIN_GROUP_N,
    min_pos: int = MIN_GROUP_POS,
) -> float | None:
    subset = table
    if levels is not None:
        subset = subset.loc[subset["group"].isin(list(levels))]
    subset = subset.loc[(subset["n"] >= min_n) & (subset["n_pos"] >= min_pos)]
    subset = subset.loc[subset["fnr"].notna()]
    if len(subset) < 2:
        return None
    return float(subset["fnr"].max() - subset["fnr"].min())


def pool_metrics(frame: pd.DataFrame, budget: float = BUDGET) -> dict:
    """Global ranking metrics plus recall under a within-module budget."""
    y = frame["at_risk"].to_numpy()
    scores = frame["score"].to_numpy()
    flag = within_module_flags(frame["code_module"].to_numpy(), scores, budget)
    global_flag = flags_at_budget(scores, budget)
    out = ranking_metrics(y, scores)
    out.update(
        {
            "recall_at_20": recall_from_flags(y, flag),
            "recall_at_20_global": recall_from_flags(y, global_flag),
            "n_test": int(len(frame)),
            "n_pos_test": int(np.asarray(y).sum()),
            "base_rate": float(np.mean(y)) if len(frame) else None,
            "selection_rate": float(flag.mean()) if len(frame) else None,
            "flag": flag,
        }
    )
    return out


def aggregate_folds(fold_rows: list[dict]) -> dict:
    """Weight PR-AUC and ROC-AUC by test size, and recall by the number of positives."""
    usable = [row for row in fold_rows if row.get("pr_auc") is not None]
    if not usable:
        return {
            "pr_auc": None,
            "roc_auc": None,
            "recall_at_20": None,
            "pr_auc_min": None,
            "pr_auc_max": None,
            "recall_at_20_min": None,
            "recall_at_20_max": None,
            "n_test": int(sum(row["n_test"] for row in fold_rows)),
            "n_pos_test": int(sum(row["n_pos_test"] for row in fold_rows)),
        }
    weight = sum(row["n_test"] for row in usable)
    positive_weight = sum(row["n_pos_test"] for row in usable if row.get("recall_at_20") is not None)
    recall_rows = [row for row in usable if row.get("recall_at_20") is not None and row["n_pos_test"]]
    return {
        "pr_auc": sum(row["pr_auc"] * row["n_test"] for row in usable) / weight,
        "roc_auc": sum(row["roc_auc"] * row["n_test"] for row in usable) / weight,
        "recall_at_20": (
            sum(row["recall_at_20"] * row["n_pos_test"] for row in recall_rows) / positive_weight
            if positive_weight
            else None
        ),
        "pr_auc_min": min(row["pr_auc"] for row in usable),
        "pr_auc_max": max(row["pr_auc"] for row in usable),
        "recall_at_20_min": min(row["recall_at_20"] for row in recall_rows) if recall_rows else None,
        "recall_at_20_max": max(row["recall_at_20"] for row in recall_rows) if recall_rows else None,
        "n_test": int(sum(row["n_test"] for row in fold_rows)),
        "n_pos_test": int(sum(row["n_pos_test"] for row in fold_rows)),
        "base_rate": (
            sum(row["n_pos_test"] for row in fold_rows) / sum(row["n_test"] for row in fold_rows)
            if sum(row["n_test"] for row in fold_rows)
            else None
        ),
    }
