"""Fit the cutoff models, write figures, and save the tables the report quotes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from oulad.config import (
    BUDGET,
    CACHE_VERSION,
    CUTOFFS,
    FIG_DIR,
    GROUP_COLUMNS,
    PROC_DIR,
    PROTECTED_FEATURES,
    PROXY_FEATURES,
    RANDOM_SEED,
    REPORT_DIR,
    TEST_PRESENTATIONS,
    TRAIN_PRESENTATIONS,
)
from oulad.eda import make_eda_figures
from oulad.fairness import (
    check_against_fairlearn,
    mitigation_curve,
    plot_ablation,
    plot_calibration,
    plot_fnr_bars,
    plot_gap_by_cutoff,
    plot_importance,
    plot_tradeoff,
    reports_for_frame,
    unmitigated_point,
)
from oulad.features import (
    LEAKAGE_RULES,
    build_features,
    exclusion_report,
    feature_columns,
    prepare,
)
from oulad.load import load_tables
from oulad.metrics import aggregate_folds, pool_metrics, within_module_flags
from oulad.modeling import (
    fit_lgbm,
    fit_logistic,
    gain_importance,
    modeling_columns,
    select_hyperparams,
    write_shap,
)
from oulad.plotting import save
from oulad.splits import (
    iter_leave_one_module_out,
    presentation_overlap,
    random_split,
    student_overlap,
    temporal_split,
    inner_temporal_split,
)

SCORE_COLUMNS = [
    "code_module",
    "code_presentation",
    "id_student",
    "at_risk",
    "final_result",
    "imd_group",
    "imd_band",
    "disability",
    "age_band",
    "gender",
]


def log(message: str) -> None:
    print(message, flush=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _context_paths() -> dict[str, Path]:
    return {
        "cohort": PROC_DIR / f"cohort_{CACHE_VERSION}.parquet",
        "daily": PROC_DIR / f"daily_{CACHE_VERSION}.parquet",
        "typed": PROC_DIR / f"typed_{CACHE_VERSION}.parquet",
        "assessments": PROC_DIR / f"assessments_{CACHE_VERSION}.parquet",
        "submissions": PROC_DIR / f"submissions_{CACHE_VERSION}.parquet",
        "prestart": PROC_DIR / f"prestart_{CACHE_VERSION}.parquet",
        "activity_types": PROC_DIR / f"activity_types_{CACHE_VERSION}.json",
    }


def save_context(ctx) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    paths = _context_paths()
    ctx.cohort.to_parquet(paths["cohort"], index=False)
    ctx.daily.to_parquet(paths["daily"], index=False)
    ctx.typed.to_parquet(paths["typed"], index=False)
    ctx.assessments.to_parquet(paths["assessments"], index=False)
    ctx.submissions.to_parquet(paths["submissions"], index=False)
    ctx.prestart.to_parquet(paths["prestart"], index=False)
    paths["activity_types"].write_text(json.dumps(ctx.activity_types))


def load_context():
    from oulad.features import Context

    paths = _context_paths()
    keys = ["code_module", "code_presentation", "id_student"]
    daily = pd.read_parquet(paths["daily"]).sort_values(keys + ["date"], kind="mergesort")
    typed = pd.read_parquet(paths["typed"]).sort_values(
        keys + ["activity_type", "date"], kind="mergesort"
    )
    return Context(
        cohort=pd.read_parquet(paths["cohort"]),
        daily=daily,
        typed=typed,
        assessments=pd.read_parquet(paths["assessments"]),
        submissions=pd.read_parquet(paths["submissions"]),
        prestart=pd.read_parquet(paths["prestart"]),
        activity_types=json.loads(paths["activity_types"].read_text()),
    )


def get_context(rebuild: bool = False):
    paths = _context_paths()
    if not rebuild and all(path.exists() for path in paths.values()):
        log("Loading cached cohort and click totals.")
        return load_context()
    log("Reading the OU csv release and building click totals.")
    tables = load_tables()
    ctx = prepare(tables)
    del tables
    save_context(ctx)
    return ctx


def feature_path(cutoff: int, active_only: bool) -> Path:
    scope = "active" if active_only else "keepwithdrawn"
    return PROC_DIR / f"features_{CACHE_VERSION}_c{cutoff}_{scope}.parquet"


def get_features(ctx, cutoff: int, active_only: bool = True, rebuild: bool = False) -> pd.DataFrame:
    path = feature_path(cutoff, active_only)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    log(f"Building features at day {cutoff} (active_only={active_only}).")
    frame = build_features(ctx, cutoff, active_only=active_only)
    frame.to_parquet(path, index=False)
    log(f"  {len(frame):,} rows, at-risk rate {frame['at_risk'].mean():.3f}.")
    return frame


def _prediction_frame(test: pd.DataFrame, scores: np.ndarray, **meta) -> pd.DataFrame:
    out = test[SCORE_COLUMNS].copy()
    out["score"] = scores
    out["n_train"] = int(meta.pop("n_train"))
    for key, value in meta.items():
        out[key] = value
    return out


def _fit_scores(train, test, columns, kind: str, params: dict, return_details: bool = False):
    if kind == "lr":
        return fit_logistic(train, test, columns)
    return fit_lgbm(train, test, columns, params, return_details=return_details)


def _summarize(frame: pd.DataFrame, split: str) -> tuple[dict, list[dict]]:
    if split == "lomo":
        fold_rows = []
        for fold, part in frame.groupby("fold"):
            stats = pool_metrics(part)
            stats.pop("flag", None)
            stats["fold"] = str(fold)
            stats["n_train"] = int(part["n_train"].iloc[0])
            fold_rows.append(stats)
        summary = aggregate_folds(fold_rows)
        summary["n_train"] = int(np.mean([row["n_train"] for row in fold_rows]))
        return summary, fold_rows
    stats = pool_metrics(frame)
    stats.pop("flag", None)
    if "n_train" in frame.columns:
        stats["n_train"] = int(frame["n_train"].iloc[0])
    return stats, []


def _score_models_for_cutoff(frame, columns, params, cutoff: int, rebuild: bool) -> list[pd.DataFrame]:
    """Temporal, random, and leave-one-module-out scores for both models."""
    pieces = []
    specs = ("temporal", "random", "lomo")
    for split_name in specs:
        for kind in ("lr", "lgbm"):
            part_path = (
                PROC_DIR / "scores" / CACHE_VERSION / f"{split_name}_{kind}_c{cutoff}.parquet"
            )
            need_details = kind == "lgbm" and split_name == "temporal" and cutoff in (0, 28)
            shap_ready = (FIG_DIR / "shap_day28.png").exists() and (
                FIG_DIR / "importance_day0_vs_day28.png"
            ).exists()
            if part_path.exists() and not rebuild and not (need_details and not shap_ready):
                pieces.append(pd.read_parquet(part_path))
                continue
            started = time.perf_counter()
            details = None
            if split_name == "temporal":
                train, test = temporal_split(frame)
                overlap = student_overlap(train, test)
                shared_presentation = presentation_overlap(train, test)
                if need_details:
                    scores, details = _fit_scores(
                        train, test, columns, kind, params, return_details=True
                    )
                else:
                    scores = _fit_scores(train, test, columns, kind, params)
                scored = _prediction_frame(
                    test,
                    scores,
                    n_train=len(train),
                    split=split_name,
                    model=kind,
                    cutoff=cutoff,
                    fold="all",
                    part="test",
                    student_overlap=overlap,
                    presentation_overlap=shared_presentation,
                )
                if details and cutoff == 28:
                    write_shap(details["model"], details["X_test"], FIG_DIR / "shap_day28.png")
                    details["importance"] = gain_importance(details["model"], details["features"])
                    details["importance"]["cutoff"] = 28
                if details and cutoff == 0:
                    details["importance"] = gain_importance(details["model"], details["features"])
                    details["importance"]["cutoff"] = 0
                if details and "importance" in details:
                    scored.attrs["importance"] = details["importance"]
            elif split_name == "random":
                train, test = random_split(frame)
                overlap = student_overlap(train, test)
                shared_presentation = presentation_overlap(train, test)
                scores = _fit_scores(train, test, columns, kind, params)
                scored = _prediction_frame(
                    test,
                    scores,
                    n_train=len(train),
                    split=split_name,
                    model=kind,
                    cutoff=cutoff,
                    fold="all",
                    part="test",
                    student_overlap=overlap,
                    presentation_overlap=shared_presentation,
                )
            else:
                fold_frames = []
                for module, train, test in iter_leave_one_module_out(frame):
                    scores = _fit_scores(train, test, columns, kind, params)
                    fold_frames.append(
                        _prediction_frame(
                            test,
                            scores,
                            n_train=len(train),
                            split=split_name,
                            model=kind,
                            cutoff=cutoff,
                            fold=module,
                            part="test",
                            student_overlap=student_overlap(train, test),
                            presentation_overlap=presentation_overlap(train, test),
                        )
                    )
                    log(f"    lomo {kind} held out {module} ({len(test):,} rows)")
                scored = pd.concat(fold_frames, ignore_index=True)
            part_path.parent.mkdir(parents=True, exist_ok=True)
            # attrs are not written to parquet; stash importance beside the file.
            importance = scored.attrs.pop("importance", None)
            if importance is not None:
                importance.to_parquet(part_path.with_suffix(".importance.parquet"), index=False)
            scored.to_parquet(part_path, index=False)
            pieces.append(scored)
            log(
                f"  {split_name} {kind} day {cutoff}: {len(scored):,} scores "
                f"in {time.perf_counter() - started:.1f}s"
            )
    return pieces


def _metrics_from_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    group_cols = ["split", "model", "cutoff"]
    for keys, part in scores.groupby(group_cols, sort=False):
        split, model, cutoff = keys
        summary, folds = _summarize(part, split)
        summary.update({"split": split, "model": model, "cutoff": int(cutoff)})
        if "student_overlap" in part.columns and split != "lomo":
            summary["student_overlap"] = float(part["student_overlap"].iloc[0])
            summary["presentation_overlap"] = float(part["presentation_overlap"].iloc[0])
        if split == "lomo" and "student_overlap" in part.columns:
            summary["student_overlap"] = float(part.groupby("fold")["student_overlap"].first().mean())
            summary["presentation_overlap"] = float(
                part.groupby("fold")["presentation_overlap"].first().mean()
            )
        rows.append(summary)
        for fold in folds:
            fold.update({"split": split, "model": model, "cutoff": int(cutoff)})
            fold_rows.append(fold)
    metrics = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    return metrics, folds


def _seen_module_rows(scores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    seen = set(
        features.loc[features["code_presentation"].isin(TRAIN_PRESENTATIONS), "code_module"].unique()
    )
    rows = []
    temporal = scores.loc[(scores["split"] == "temporal") & (scores["model"] == "lgbm")]
    for cutoff, part in temporal.groupby("cutoff"):
        for name, mask in (
            ("temporal_seen_modules", part["code_module"].isin(seen)),
            ("temporal_unseen_modules", ~part["code_module"].isin(seen)),
        ):
            subset = part.loc[mask]
            if len(subset) < 20 or subset["at_risk"].nunique() < 2:
                continue
            stats = pool_metrics(subset)
            stats.pop("flag", None)
            stats.update(
                {
                    "split": name,
                    "model": "lgbm",
                    "cutoff": int(cutoff),
                    "n_train": int(part["n_train"].iloc[0]),
                }
            )
            rows.append(stats)
    return pd.DataFrame(rows)


def plot_metric_curves(metrics: pd.DataFrame, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    colors = {"temporal": "#0072B2", "random": "#E69F00", "lomo": "#009E73"}
    labels = {
        "temporal": "Temporal (2013 to 2014)",
        "random": "Random row split",
        "lomo": "Leave-one-module-out",
    }
    subset = metrics.loc[(metrics["model"] == "lgbm") & (metrics["split"].isin(colors))]
    for split, color in colors.items():
        part = subset.loc[subset["split"] == split].sort_values("cutoff")
        axes[0].plot(part["cutoff"], part["pr_auc"], marker="o", color=color, label=labels[split])
        axes[1].plot(
            part["cutoff"], part["recall_at_20"], marker="o", color=color, label=labels[split]
        )
        if split == "lomo" and "pr_auc_min" in part.columns:
            axes[0].fill_between(
                part["cutoff"], part["pr_auc_min"], part["pr_auc_max"], color=color, alpha=0.15
            )
    temporal = subset.loc[subset["split"] == "temporal"].sort_values("cutoff")
    if len(temporal):
        axes[0].plot(
            temporal["cutoff"],
            temporal["base_rate"],
            linestyle="--",
            color="#444444",
            label="Temporal base rate",
        )
    axes[1].axhline(BUDGET, color="#444444", linestyle="--", label="Random 20% list")
    axes[0].set_ylabel("PR-AUC")
    axes[1].set_ylabel("Recall at a 20% within-module budget")
    for axis in axes:
        axis.set_xlabel("Cutoff day")
        axis.set_xticks(list(CUTOFFS))
        axis.legend(fontsize=8)
    axes[0].set_title("Ranking")
    axes[1].set_title("Outreach list")
    figure.suptitle("LightGBM, three ways of holding out students", fontsize=13)
    figure.tight_layout()
    save(figure, path)
    plt.close(figure)


def plot_model_comparison(metrics: pd.DataFrame, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    colors = {"lr": "#CC79A7", "lgbm": "#0072B2"}
    names = {"lr": "Logistic regression", "lgbm": "LightGBM"}
    temporal = metrics.loc[metrics["split"] == "temporal"]
    for kind, color in colors.items():
        part = temporal.loc[temporal["model"] == kind].sort_values("cutoff")
        axes[0].plot(part["cutoff"], part["pr_auc"], marker="o", color=color, label=names[kind])
        axes[1].plot(
            part["cutoff"], part["recall_at_20"], marker="o", color=color, label=names[kind]
        )
    axes[1].axhline(BUDGET, color="#444444", linestyle="--", label="Random 20% list")
    axes[0].set_ylabel("PR-AUC")
    axes[1].set_ylabel("Recall at 20% within module")
    for axis in axes:
        axis.set_xlabel("Cutoff day")
        axis.set_xticks(list(CUTOFFS))
        axis.legend(fontsize=8)
    figure.suptitle("Temporal split: baseline versus boosted trees", fontsize=13)
    figure.tight_layout()
    save(figure, path)
    plt.close(figure)


def plot_lomo_modules(folds: pd.DataFrame, path) -> None:
    part = folds.loc[
        (folds["split"] == "lomo") & (folds["model"] == "lgbm") & (folds["cutoff"] == 28)
    ].sort_values("fold")
    if part.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].barh(part["fold"], part["pr_auc"], color="#009E73")
    axes[1].barh(part["fold"], part["recall_at_20"], color="#0072B2")
    axes[0].set_xlabel("PR-AUC")
    axes[1].set_xlabel("Recall at 20%")
    axes[0].set_title("Day 28, module held out")
    axes[1].set_title("Day 28 outreach recall")
    figure.suptitle("Leave-one-module-out is not one number", fontsize=13)
    figure.tight_layout()
    save(figure, path)
    plt.close(figure)


def _load_importance(cutoff: int) -> pd.DataFrame | None:
    path = PROC_DIR / "scores" / CACHE_VERSION / f"temporal_lgbm_c{cutoff}.importance.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _ablation_and_sensitivity(features_by_cutoff, params, rebuild: bool) -> pd.DataFrame:
    pieces = []
    jobs = [
        (28, "temporal_keep_withdrawn", "keep", ()),
        (28, "ablate_protected", "active", PROTECTED_FEATURES),
        (56, "ablate_protected", "active", PROTECTED_FEATURES),
        (28, "ablate_proxies", "active", PROTECTED_FEATURES + PROXY_FEATURES),
        (56, "ablate_proxies", "active", PROTECTED_FEATURES + PROXY_FEATURES),
    ]
    for cutoff, split_name, _scope, drop in jobs:
        part_path = PROC_DIR / "scores" / CACHE_VERSION / f"{split_name}_lgbm_c{cutoff}.parquet"
        if part_path.exists() and not rebuild:
            pieces.append(pd.read_parquet(part_path))
            continue
        frame = features_by_cutoff[(cutoff, split_name == "temporal_keep_withdrawn")]
        columns = modeling_columns(frame, drop=drop)
        train, test = temporal_split(frame)
        scores = fit_lgbm(train, test, columns, params)
        scored = _prediction_frame(
            test,
            scores,
            n_train=len(train),
            split=split_name,
            model="lgbm",
            cutoff=cutoff,
            fold="all",
            part="test",
        )
        part_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(part_path, index=False)
        pieces.append(scored)
        log(f"  extra fit {split_name} day {cutoff}")
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _threshold_frames(frame: pd.DataFrame, params: dict, cutoff: int, rebuild: bool):
    path = PROC_DIR / "scores" / CACHE_VERSION / f"threshold_lgbm_c{cutoff}.parquet"
    if path.exists() and not rebuild:
        saved = pd.read_parquet(path)
        return saved.loc[saved["part"] == "val"], saved.loc[saved["part"] == "test"]
    pool = frame.loc[frame["code_presentation"].isin(TRAIN_PRESENTATIONS)].reset_index(drop=True)
    test = frame.loc[frame["code_presentation"].isin(TEST_PRESENTATIONS)].reset_index(drop=True)
    train_idx, val_idx = train_test_split(
        pool.index,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=pool["at_risk"].to_numpy(),
    )
    train = pool.loc[train_idx]
    validation = pool.loc[val_idx]
    columns = modeling_columns(frame)
    # One fit, two matrices: call twice so each score vector matches its rows.
    # Refitting would change the model. Score both by stacking and splitting.
    combined = pd.concat([validation, test], ignore_index=True)
    scores = fit_lgbm(train, combined, columns, params)
    n_val = len(validation)
    val_scored = _prediction_frame(
        validation,
        scores[:n_val],
        n_train=len(train),
        split="threshold",
        model="lgbm",
        cutoff=cutoff,
        fold="all",
        part="val",
    )
    test_scored = _prediction_frame(
        test,
        scores[n_val:],
        n_train=len(train),
        split="threshold",
        model="lgbm",
        cutoff=cutoff,
        fold="all",
        part="test",
    )
    saved = pd.concat([val_scored, test_scored], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved.to_parquet(path, index=False)
    return val_scored, test_scored


def run_fairness(scores: pd.DataFrame, features_by_cutoff: dict, params: dict, rebuild: bool) -> dict:
    temporal = scores.loc[(scores["split"] == "temporal") & (scores["model"] == "lgbm")].copy()
    tables = []
    gap_rows = []
    for cutoff, part in temporal.groupby("cutoff"):
        report = reports_for_frame(part)
        report["cutoff"] = int(cutoff)
        tables.append(report)
        if int(cutoff) == 28:
            flags = within_module_flags(part["code_module"].to_numpy(), part["score"].to_numpy())
            check_against_fairlearn(part["at_risk"], flags, part["imd_group"])
            log("fairlearn agrees with the local FNR definition on the day-28 IMD split.")
        for attribute in GROUP_COLUMNS:
            gap = report.loc[report["attribute"] == attribute, "fnr_gap"]
            gap_rows.append(
                {
                    "cutoff": int(cutoff),
                    "attribute": attribute,
                    "fnr_gap": None if gap.empty else gap.iloc[0],
                }
            )
    group_metrics = pd.concat(tables, ignore_index=True)
    gaps = pd.DataFrame(gap_rows)
    plot_fnr_bars(
        group_metrics.loc[group_metrics["cutoff"] == 28],
        FIG_DIR / "fairness_fnr_day28.png",
        "Day 28, missed at-risk students by group (grey bars are too small for the gap)",
    )
    plot_fnr_bars(
        group_metrics.loc[group_metrics["cutoff"] == 56],
        FIG_DIR / "fairness_fnr_day56.png",
        "Day 56, missed at-risk students by group",
    )
    plot_gap_by_cutoff(gaps, FIG_DIR / "fnr_gap_vs_cutoff.png")
    day28 = temporal.loc[temporal["cutoff"] == 28]
    plot_calibration(day28, FIG_DIR / "calibration_imd_day28.png")
    plt.close("all")

    extra = _ablation_and_sensitivity(features_by_cutoff, params, rebuild)
    ablation_rows = []
    # The full model is the temporal fit already stored.
    for cutoff in (28, 56):
        full = temporal.loc[temporal["cutoff"] == cutoff]
        variants = {
            "full": full,
            "drop protected": extra.loc[
                (extra["split"] == "ablate_protected") & (extra["cutoff"] == cutoff)
            ],
            "drop protected and proxies": extra.loc[
                (extra["split"] == "ablate_proxies") & (extra["cutoff"] == cutoff)
            ],
        }
        for variant, part in variants.items():
            if part.empty:
                continue
            stats = pool_metrics(part)
            report = reports_for_frame(part)
            for attribute in GROUP_COLUMNS:
                gap = report.loc[report["attribute"] == attribute, "fnr_gap"]
                ablation_rows.append(
                    {
                        "cutoff": cutoff,
                        "variant": variant,
                        "attribute": attribute,
                        "fnr_gap": None if gap.empty else gap.iloc[0],
                        "pr_auc": stats["pr_auc"],
                        "recall_at_20": stats["recall_at_20"],
                    }
                )
    ablation = pd.DataFrame(ablation_rows)
    if not ablation.empty:
        plot_ablation(ablation, FIG_DIR / "ablation_fnr_gap.png")
        plt.close("all")

    curves = []
    baselines = []
    for cutoff in (28, 56):
        validation, test = _threshold_frames(features_by_cutoff[(cutoff, False)], params, cutoff, rebuild)
        for attribute in GROUP_COLUMNS:
            curve = mitigation_curve(validation, test, attribute)
            curve["cutoff"] = cutoff
            curves.append(curve)
            base = unmitigated_point(test, attribute)
            base["cutoff"] = cutoff
            baselines.append(base)
        log(f"  mitigation curve day {cutoff}")
    curve_frame = pd.concat(curves, ignore_index=True)
    baseline_frame = pd.DataFrame(baselines)
    plot_tradeoff(curve_frame, baseline_frame, FIG_DIR / "mitigation_tradeoff.png")
    plt.close("all")
    return {
        "groups": group_metrics,
        "gaps": gaps,
        "ablation": ablation,
        "curves": curve_frame,
        "baselines": baseline_frame,
        "extra_scores": extra,
    }


def _summary_text(payload: dict) -> str:
    lines = ["OULAD early-warning summary", ""]
    exclusion = payload["exclusion"]
    lines.append(
        f"Registrations {exclusion['n_rows']}, never started {exclusion['n_never_started']}."
    )
    lines.append("LightGBM metrics (PR-AUC / recall@20%):")
    for row in payload["metrics"]:
        if row.get("model") != "lgbm":
            continue
        if row.get("split") not in {
            "temporal",
            "random",
            "lomo",
            "temporal_seen_modules",
            "temporal_unseen_modules",
            "temporal_keep_withdrawn",
        }:
            continue
        lines.append(
            f"  day {row['cutoff']:>3} {row['split']:<28} "
            f"PR {row.get('pr_auc')}  recall {row.get('recall_at_20')}  "
            f"n {row.get('n_test')}  base {row.get('base_rate')}"
        )
    lines.append("")
    lines.append("FNR gaps:")
    for row in payload.get("fairness_gaps", []):
        lines.append(f"  day {row['cutoff']:>3} {row['attribute']:<12} {row['fnr_gap']}")
    lines.append("")
    lines.append("Chosen mitigation points:")
    for row in payload.get("mitigation_chosen", []):
        lines.append(
            f"  day {row['cutoff']} {row['attribute']}: recall {row.get('test_recall')} "
            f"gap {row.get('fnr_gap')} selection {row.get('selection_rate')}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the OULAD early-warning analysis.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore feature and score caches.")
    parser.add_argument(
        "--rebuild-context",
        action="store_true",
        help="Reread the CSVs and rebuild click totals.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "eda", "models", "fairness"],
    )
    args = parser.parse_args(argv)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = get_context(rebuild=args.rebuild_context or args.rebuild)
    exclusion = exclusion_report(ctx.cohort)
    _write_json(REPORT_DIR / "exclusion.json", exclusion)
    log(f"Never started: {exclusion['n_never_started']}. Eligible at day 28: {exclusion['n_eligible_day_28']}.")
    if args.stage in {"all", "eda"}:
        log("Writing EDA figures.")
        make_eda_figures(ctx, FIG_DIR)
        plt.close("all")
        if args.stage == "eda":
            return

    features = {cutoff: get_features(ctx, cutoff, True, args.rebuild) for cutoff in CUTOFFS}
    features[(28, True)] = features[28]
    features[(28, False)] = get_features(ctx, 28, False, args.rebuild)
    features[(56, False)] = features[56]
    # _ablation_and_sensitivity keys (cutoff, keep_withdrawn_bool)
    feature_sets = {(cutoff, False): features[cutoff] for cutoff in CUTOFFS}
    feature_sets[(28, True)] = features[(28, False)]

    hyper_path = REPORT_DIR / "hyperparams.json"
    if hyper_path.exists() and not args.rebuild:
        best = json.loads(hyper_path.read_text())
        log(f"Reusing hyperparameters {best}.")
    else:
        log("Selecting LightGBM hyperparameters on 2013B/2013J at day 28.")
        inner_train, inner_val = inner_temporal_split(features[28])
        best = select_hyperparams(inner_train, inner_val, modeling_columns(features[28]))
        _write_json(hyper_path, best)
    params = {
        "num_leaves": best["num_leaves"],
        "learning_rate": best["learning_rate"],
        "n_estimators": best["n_estimators"],
    }

    pieces = []
    if args.stage in {"all", "models", "fairness"}:
        score_dir = PROC_DIR / "scores" / CACHE_VERSION
        existing = list(score_dir.glob("*_c*.parquet")) if score_dir.exists() else []
        # Main splits are refit or loaded inside the cutoff loop. Fairness-only
        # loads whatever parts already exist and then fills gaps.
        if args.stage == "fairness" and existing and not args.rebuild:
            log("Loading saved scores.")
            for path in existing:
                if path.name.startswith("threshold") or "importance" in path.name:
                    continue
                if path.name.startswith("ablate") or path.name.startswith("temporal_keep"):
                    continue
                frame = pd.read_parquet(path)
                if {"split", "model", "cutoff", "score"}.issubset(frame.columns):
                    pieces.append(frame)
        else:
            for cutoff in CUTOFFS:
                columns = modeling_columns(features[cutoff])
                pieces.extend(_score_models_for_cutoff(features[cutoff], columns, params, cutoff, args.rebuild))
    scores = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if scores.empty:
        raise RuntimeError("No scores were produced.")
    metrics, folds = _metrics_from_scores(scores)
    extra_rows = _seen_module_rows(scores, features[28])
    if len(extra_rows):
        metrics = pd.concat([metrics, extra_rows], ignore_index=True)

    if args.stage in {"all", "models"}:
        plot_metric_curves(metrics, FIG_DIR / "metric_vs_cutoff.png")
        plot_model_comparison(metrics, FIG_DIR / "model_comparison.png")
        plot_lomo_modules(folds, FIG_DIR / "lomo_by_module.png")
        day0 = _load_importance(0)
        day28 = _load_importance(28)
        if day0 is not None and day28 is not None:
            plot_importance(day0, day28, FIG_DIR / "importance_day0_vs_day28.png")
            plt.close("all")

    fairness = run_fairness(scores, feature_sets, params, args.rebuild)
    if len(fairness["extra_scores"]):
        extra_metrics, _ = _metrics_from_scores(fairness["extra_scores"])
        metrics = pd.concat([metrics, extra_metrics], ignore_index=True)

    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    if len(folds):
        folds.to_csv(REPORT_DIR / "lomo_modules.csv", index=False)
    fairness["groups"].to_csv(REPORT_DIR / "fairness_groups.csv", index=False)
    fairness["ablation"].to_csv(REPORT_DIR / "ablation.csv", index=False)
    fairness["curves"].to_csv(REPORT_DIR / "mitigation.csv", index=False)
    chosen = fairness["curves"].loc[fairness["curves"]["chosen"]].copy()
    payload = {
        "hyperparams": best,
        "exclusion": exclusion,
        "leakage_rules": list(LEAKAGE_RULES),
        "budget": BUDGET,
        "metrics": metrics.to_dict(orient="records"),
        "fairness_gaps": fairness["gaps"].to_dict(orient="records"),
        "ablation": fairness["ablation"].to_dict(orient="records"),
        "mitigation_chosen": chosen.to_dict(orient="records"),
        "mitigation_baselines": fairness["baselines"].to_dict(orient="records"),
    }
    _write_json(REPORT_DIR / "results.json", payload)
    text = _summary_text(payload)
    (REPORT_DIR / "summary.txt").write_text(text)
    log(text)
    log(f"Finished in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
