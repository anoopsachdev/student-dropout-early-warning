"""Group error rates at a fixed outreach budget, and a threshold adjustment.

The harm that matters here is a missed at-risk student: the false-negative rate
among people who withdraw or fail. Selection is the within-module top 20%, so a
hard module cannot consume the whole university's outreach list.

The mitigation does not retrain the booster. On a held-out slice of 2013 it
picks a score threshold per group so that each group has the same recall, then
applies those thresholds to 2014. Sweeping the target recall traces the
tradeoff against the gap in false-negative rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from oulad.config import GROUP_COLUMNS, IMD_GROUP_ORDER, MIN_GROUP_N, MIN_GROUP_POS
from oulad.metrics import (
    fnr_gap,
    group_table,
    rates,
    recall_from_flags,
    within_module_flags,
)
from oulad.plotting import save

GAP_LEVELS = {
    "imd_group": ("low", "mid", "high"),
    "disability": None,
    "age_band": None,
    "gender": None,
}


def fairlearn_table(y, flag, groups) -> pd.DataFrame:
    """Same rates via fairlearn, used as a check on the local definitions."""
    from fairlearn.metrics import (
        MetricFrame,
        false_negative_rate,
        false_positive_rate,
        selection_rate,
    )

    frame = MetricFrame(
        metrics={
            "fnr": false_negative_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=np.asarray(y).astype(int),
        y_pred=np.asarray(flag).astype(int),
        sensitive_features=np.asarray(groups),
    )
    out = frame.by_group.reset_index()
    group_name = "sensitive_feature_0" if "sensitive_feature_0" in out.columns else out.columns[0]
    out = out.rename(columns={group_name: "group"})
    return out


def check_against_fairlearn(y, flag, groups) -> None:
    local = group_table(y, flag, groups)
    other = fairlearn_table(y, flag, groups)
    merged = local.merge(other, on="group", suffixes=("_local", "_fairlearn"))
    for metric in ("fnr", "fpr", "selection_rate"):
        left = merged[f"{metric}_local"].to_numpy(dtype=float)
        right = merged[f"{metric}_fairlearn"].to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True, atol=1e-6):
            raise AssertionError(f"fairlearn {metric} disagrees with the local definition.")


def group_report(frame: pd.DataFrame, attribute: str, budget_flags: np.ndarray) -> pd.DataFrame:
    table = group_table(frame["at_risk"].to_numpy(), budget_flags, frame[attribute].to_numpy())
    table.insert(0, "attribute", attribute)
    levels = GAP_LEVELS[attribute]
    gap = fnr_gap(table, levels=levels)
    table["fnr_gap"] = gap
    table["in_gap"] = False
    eligible = table["n"] >= MIN_GROUP_N
    eligible &= table["n_pos"] >= MIN_GROUP_POS
    eligible &= table["fnr"].notna()
    if levels is not None:
        eligible &= table["group"].isin(list(levels))
    if int(eligible.sum()) >= 2:
        table.loc[eligible, "in_gap"] = True
    return table


def reports_for_frame(frame: pd.DataFrame) -> pd.DataFrame:
    flags = within_module_flags(frame["code_module"].to_numpy(), frame["score"].to_numpy())
    tables = [group_report(frame, attribute, flags) for attribute in GROUP_COLUMNS]
    return pd.concat(tables, ignore_index=True)


def thresholds_for_recall(
    y, scores, groups, target_recall: float, min_positives: int = MIN_GROUP_POS
) -> dict[str, dict]:
    """Score cutoff per group so about ``target_recall`` of validation positives are flagged."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)
    positives = scores[y == 1]
    if len(positives) == 0:
        raise ValueError("Cannot set recall thresholds without positive cases.")
    global_threshold = float(np.quantile(positives, 1.0 - target_recall))
    chosen = {}
    for group in pd.unique(groups):
        group_scores = scores[(groups == group) & (y == 1)]
        fallback = len(group_scores) < min_positives
        threshold = global_threshold if fallback else float(np.quantile(group_scores, 1.0 - target_recall))
        chosen[str(group)] = {
            "threshold": threshold,
            "n_pos": int(len(group_scores)),
            "fallback": fallback,
        }
    return chosen


def apply_group_thresholds(scores, groups, chosen: dict[str, dict]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)
    flag = np.zeros(len(scores), dtype=bool)
    for group, spec in chosen.items():
        flag[(groups == group) & (scores >= spec["threshold"])] = True
    return flag


def mitigation_curve(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    attribute: str,
    targets: np.ndarray | None = None,
) -> pd.DataFrame:
    if targets is None:
        targets = np.round(np.arange(0.15, 0.86, 0.05), 2)
    levels = GAP_LEVELS[attribute]
    rows = []
    for target in targets:
        chosen = thresholds_for_recall(
            validation["at_risk"], validation["score"], validation[attribute], float(target)
        )
        flag = apply_group_thresholds(test["score"], test[attribute], chosen)
        table = group_table(test["at_risk"], flag, test[attribute])
        overall = rates(test["at_risk"], flag)
        rows.append(
            {
                "attribute": attribute,
                "target_recall": float(target),
                "test_recall": overall["recall"],
                "selection_rate": overall["selection_rate"],
                "fnr_gap": fnr_gap(table, levels=levels),
                "n_fallback_groups": int(sum(spec["fallback"] for spec in chosen.values())),
            }
        )
    curve = pd.DataFrame(rows)
    band = curve["selection_rate"].between(0.15, 0.25) & curve["fnr_gap"].notna()
    curve["chosen"] = False
    if band.any():
        pick = curve.loc[band, "fnr_gap"].idxmin()
    elif curve["fnr_gap"].notna().any():
        pick = (curve["selection_rate"] - 0.20).abs().idxmin()
    else:
        pick = None
    if pick is not None:
        curve.loc[pick, "chosen"] = True
    return curve


def unmitigated_point(test: pd.DataFrame, attribute: str) -> dict:
    flag = within_module_flags(test["code_module"].to_numpy(), test["score"].to_numpy())
    table = group_table(test["at_risk"], flag, test[attribute])
    return {
        "attribute": attribute,
        "test_recall": recall_from_flags(test["at_risk"], flag),
        "selection_rate": float(flag.mean()),
        "fnr_gap": fnr_gap(table, levels=GAP_LEVELS[attribute]),
    }


def plot_fnr_bars(table: pd.DataFrame, path, title: str):
    import matplotlib.pyplot as plt

    attributes = [attribute for attribute in GROUP_COLUMNS if attribute in set(table["attribute"])]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for axis, attribute in zip(axes.ravel(), attributes):
        part = table.loc[table["attribute"] == attribute].copy()
        if attribute == "imd_group":
            order = [group for group in IMD_GROUP_ORDER if group in set(part["group"])]
            part = part.set_index("group").loc[order].reset_index()
        else:
            part = part.sort_values("group")
        colors = ["#0072B2" if flag else "#B0B0B0" for flag in part["in_gap"]]
        axis.barh(part["group"].astype(str), part["fnr"], color=colors)
        axis.set_xlim(0, 1)
        axis.set_xlabel("False-negative rate")
        gap = part["fnr_gap"].iloc[0] if len(part) else None
        gap_text = "n/a" if gap is None or (isinstance(gap, float) and np.isnan(gap)) else f"{gap:.2f}"
        axis.set_title(f"{attribute}  (FNR gap {gap_text})")
    figure.suptitle(title, fontsize=13, fontweight="semibold")
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_gap_by_cutoff(gap_frame: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for color, attribute in zip(colors, GROUP_COLUMNS):
        part = gap_frame.loc[gap_frame["attribute"] == attribute].sort_values("cutoff")
        axis.plot(part["cutoff"], part["fnr_gap"], marker="o", color=color, label=attribute)
    axis.set_xlabel("Cutoff day")
    axis.set_ylabel("FNR gap at a 20% within-module budget")
    finite = gap_frame["fnr_gap"].dropna()
    highest = float(finite.max()) if len(finite) else 0.2
    axis.set_ylim(0, max(0.2, highest * 1.15))
    axis.set_title("Does the disparity shrink once behaviour is observed?")
    axis.legend(ncol=2)
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_calibration(frame: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="perfect")
    colors = {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2", "Missing": "#999999"}
    for group, color in colors.items():
        part = frame.loc[frame["imd_group"] == group]
        if len(part) < 200 or part["at_risk"].nunique() < 2:
            continue
        fraction, mean_score = calibration_curve(
            part["at_risk"], part["score"], n_bins=8, strategy="quantile"
        )
        axis.plot(mean_score, fraction, marker="o", color=color, label=group)
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Observed at-risk rate")
    axis.set_title("Calibration by deprivation band, day 28")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_ablation(ablation: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    attributes = list(GROUP_COLUMNS)
    variants = ["full", "drop protected", "drop protected and proxies"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    for axis, cutoff in zip(axes, (28, 56)):
        part = ablation.loc[ablation["cutoff"] == cutoff]
        index = np.arange(len(attributes))
        width = 0.24
        for offset, variant in enumerate(variants):
            gaps = []
            for attribute in attributes:
                row = part.loc[(part["attribute"] == attribute) & (part["variant"] == variant)]
                gaps.append(float(row["fnr_gap"].iloc[0]) if len(row) else np.nan)
            axis.bar(index + (offset - 1) * width, gaps, width=width, label=variant)
        axis.set_xticks(index, attributes, rotation=15)
        axis.set_title(f"Day {cutoff}")
        axis.set_ylabel("FNR gap" if cutoff == 28 else "")
    axes[0].legend(fontsize=9)
    figure.suptitle("Dropping the attributes shrinks some gaps and leaves others", fontsize=13)
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_tradeoff(curves: pd.DataFrame, baselines: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    colors = {
        "imd_group": "#D55E00",
        "disability": "#0072B2",
        "age_band": "#009E73",
        "gender": "#CC79A7",
    }
    for axis, cutoff in zip(axes, sorted(curves["cutoff"].unique())):
        for attribute, color in colors.items():
            part = curves.loc[
                (curves["cutoff"] == cutoff) & (curves["attribute"] == attribute)
            ].dropna(subset=["test_recall", "fnr_gap"]).sort_values("test_recall")
            axis.plot(
                part["test_recall"],
                part["fnr_gap"],
                marker="o",
                color=color,
                label=attribute,
                linewidth=1.4,
            )
            base = baselines.loc[
                (baselines["cutoff"] == cutoff) & (baselines["attribute"] == attribute)
            ]
            if len(base):
                axis.scatter(
                    base["test_recall"],
                    base["fnr_gap"],
                    marker="*",
                    s=160,
                    color=color,
                    zorder=3,
                )
        axis.set_xlabel("Recall on 2014")
        axis.set_title(f"Day {cutoff}")
        axis.set_xlim(0, 1)
    axes[0].set_ylabel("FNR gap")
    axes[0].legend(fontsize=9)
    figure.suptitle(
        "Equalising recall across groups (lines) versus a shared within-module top 20% (stars)",
        fontsize=12,
    )
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_importance(day0: pd.DataFrame, day28: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    from oulad.features import family_of

    families = ["protected", "proxy", "other static", "engagement", "assessment"]
    shares = []
    for frame in (day0, day28):
        tagged = frame.copy()
        tagged["family"] = tagged["feature"].map(family_of)
        total = tagged.groupby("family")["share"].sum()
        shares.append([float(total.get(family, 0.0)) for family in families])
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    index = np.arange(len(families))
    axis.barh(index - 0.18, shares[0], height=0.36, color="#E69F00", label="Day 0")
    axis.barh(index + 0.18, shares[1], height=0.36, color="#0072B2", label="Day 28")
    axis.set_yticks(index, families)
    axis.set_xlabel("Share of total split gain")
    axis.set_title("Day 0 is early clicks and background; day 28 is assessment")
    axis.legend()
    figure.tight_layout()
    save(figure, path)
    return figure


