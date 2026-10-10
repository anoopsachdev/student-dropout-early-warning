"""Descriptive figures. None of these are fed to the early-warning model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from oulad.config import IMD_ORDER, KEYS, OUTCOME_COLORS, OUTCOME_ORDER
from oulad.features import Context
from oulad.plotting import save


def _started(cohort: pd.DataFrame) -> pd.DataFrame:
    unreg = cohort["date_unregistration"]
    return cohort.loc[~(unreg.notna() & (unreg < 0))].copy()


def _outcome_share(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    counts = pd.crosstab(frame[by], frame["final_result"])
    for outcome in OUTCOME_ORDER:
        if outcome not in counts.columns:
            counts[outcome] = 0
    counts = counts[list(OUTCOME_ORDER)]
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    shares["n"] = counts.sum(axis=1)
    return shares


def plot_outcome_mix(cohort: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    frame = cohort.copy()
    frame["presentation"] = frame["code_module"] + " " + frame["code_presentation"]
    order = (
        frame.groupby(["code_module", "code_presentation"], sort=True)
        .size()
        .reset_index()[["code_module", "code_presentation"]]
    )
    labels = (order["code_module"] + " " + order["code_presentation"]).tolist()
    shares = _outcome_share(frame, "presentation").reindex(labels)
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    left = np.zeros(len(shares))
    y = np.arange(len(shares))
    for outcome in OUTCOME_ORDER:
        width = shares[outcome].to_numpy()
        axis.barh(y, width, left=left, color=OUTCOME_COLORS[outcome], label=outcome)
        left = left + width
    axis.set_yticks(y, [f"{label}  n={int(n)}" for label, n in zip(labels, shares["n"])])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of registrations")
    axis.set_title("Final result by module and presentation")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.12)
    save(figure, path)
    return figure


def plot_weekly_clicks(ctx: Context, path):
    import matplotlib.pyplot as plt

    students = _started(ctx.cohort)[KEYS + ["final_result", "date_unregistration"]]
    daily = ctx.daily.loc[(ctx.daily["date"] >= 0) & (ctx.daily["date"] < 84), KEYS + ["date", "clicks"]].copy()
    daily["week"] = (daily["date"] // 7).astype(int)
    weekly = daily.groupby(KEYS + ["week"], sort=False)["clicks"].sum().reset_index()
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for outcome in OUTCOME_ORDER:
        means = []
        for week in range(12):
            start = week * 7
            still = students["date_unregistration"].isna() | (students["date_unregistration"] > start)
            pool = students.loc[still & (students["final_result"] == outcome), KEYS]
            clicks = pool.merge(weekly.loc[weekly["week"] == week, KEYS + ["clicks"]], on=KEYS, how="left")
            means.append(float(clicks["clicks"].fillna(0).mean()) if len(clicks) else np.nan)
        axis.plot(np.arange(12), means, marker="o", color=OUTCOME_COLORS[outcome], label=outcome)
    axis.set_xlabel("Week of the presentation")
    axis.set_ylabel("Mean clicks, students still registered")
    axis.set_title("Withdrawn students are already quieter while enrolled")
    axis.legend(ncol=2)
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_first_tma(ctx: Context, path):
    import matplotlib.pyplot as plt

    tmas = ctx.assessments.loc[
        (ctx.assessments["assessment_type"].str.lower() == "tma") & ctx.assessments["date_due"].notna()
    ].sort_values(["code_module", "code_presentation", "date_due", "id_assessment"], kind="mergesort")
    first = tmas.groupby(["code_module", "code_presentation"], sort=False).head(1)
    students = _started(ctx.cohort)[KEYS + ["final_result"]]
    submitted = ctx.submissions.loc[ctx.submissions["id_assessment"].isin(first["id_assessment"])]
    submitted = submitted.sort_values("date_submitted", kind="mergesort").drop_duplicates(KEYS, keep="first")
    merged = students.merge(
        submitted[KEYS + ["score"]],
        on=KEYS,
        how="left",
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    boxes, labels = [], []
    for outcome in OUTCOME_ORDER:
        scores = merged.loc[
            (merged["final_result"] == outcome) & merged["score"].notna(), "score"
        ].to_numpy()
        if len(scores):
            boxes.append(scores)
            labels.append(outcome)
    tick = axes[0].boxplot(boxes, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(tick["boxes"], labels):
        patch.set_facecolor(OUTCOME_COLORS[label])
        patch.set_alpha(0.85)
    axes[0].axhline(40, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("First TMA, students who submitted")
    missing = 1 - merged.groupby("final_result")["score"].apply(lambda s: s.notna().mean())
    missing = missing.reindex(list(OUTCOME_ORDER))
    axes[1].bar(
        list(OUTCOME_ORDER),
        missing.to_numpy(),
        color=[OUTCOME_COLORS[outcome] for outcome in OUTCOME_ORDER],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share with no first-TMA submission")
    axes[1].set_title("Who never submits it")
    figure.tight_layout()
    save(figure, path)
    return figure


def plot_demographics(cohort: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    frame = _started(cohort)
    specs = (
        ("imd_band", IMD_ORDER, "Deprivation band"),
        ("disability", ["N", "Y", "Missing"], "Declared disability"),
        ("age_band", ["0-35", "35-55", "55<=", "Missing"], "Age band"),
        ("gender", ["F", "M", "Missing"], "Gender"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, order, title) in zip(axes.ravel(), specs):
        present = [level for level in order if level in set(frame[column])]
        extra = [level for level in frame[column].dropna().unique() if level not in present]
        present.extend(sorted(extra))
        shares = _outcome_share(frame, column).reindex(present)
        left = np.zeros(len(shares))
        y = np.arange(len(shares))
        for outcome in OUTCOME_ORDER:
            width = shares[outcome].to_numpy()
            axis.barh(y, width, left=left, color=OUTCOME_COLORS[outcome], label=outcome)
            left = left + width
        axis.set_yticks(y, [f"{level}  n={int(n)}" for level, n in zip(present, shares["n"])])
        axis.set_xlim(0, 1)
        axis.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Outcome mix among students who started", fontsize=13, y=1.06)
    figure.tight_layout()
    save(figure, path)
    return figure


def _kaplan_meier(times: np.ndarray, events: np.ndarray):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    n = len(times)
    survival = 1.0
    xs = [0.0]
    ys = [1.0]
    index = 0
    while index < n:
        moment = times[index]
        at_risk = n - index
        deaths = 0
        cursor = index
        while cursor < n and times[cursor] == moment:
            if events[cursor]:
                deaths += 1
            cursor += 1
        if deaths and at_risk:
            survival *= 1.0 - deaths / at_risk
            xs.append(float(moment))
            ys.append(survival)
        index = cursor
    if times.size and xs[-1] < times.max():
        xs.append(float(times.max()))
        ys.append(ys[-1])
    return np.asarray(xs), np.asarray(ys)


def plot_withdrawal_km(cohort: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    frame = _started(cohort)
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    colors = {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2", "Missing": "#777777"}
    event = frame["date_unregistration"].notna() & (frame["date_unregistration"] >= 0)
    time = np.where(event, frame["date_unregistration"], frame["module_length"])
    time = np.clip(time, 0, None)
    xs, ys = _kaplan_meier(time, event.to_numpy())
    axis.step(xs, ys, where="post", color="#222222", linewidth=2.0, label="All who started")
    for group, color in colors.items():
        part = frame.loc[frame["imd_group"] == group]
        if len(part) < 30:
            continue
        part_event = part["date_unregistration"].notna() & (part["date_unregistration"] >= 0)
        part_time = np.where(part_event, part["date_unregistration"], part["module_length"])
        xs, ys = _kaplan_meier(np.clip(part_time, 0, None), part_event.to_numpy())
        axis.step(xs, ys, where="post", color=color, label=group)
    axis.set_xlim(0, 280)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Day from presentation start")
    axis.set_ylabel("Still registered")
    axis.set_title("Time to withdrawal")
    axis.legend(ncol=2)
    figure.tight_layout()
    save(figure, path)
    return figure


def _cluster_labels(profiles: np.ndarray, mean_clicks: np.ndarray) -> list[str]:
    """Name each profile from its raw weekly means. Normalised shapes are all early-peaked."""
    raw = []
    for profile in profiles:
        early = float(np.mean(profile[:4]))
        mid = float(np.mean(profile[4:8]))
        late = float(np.mean(profile[8:]))
        if early <= 1:
            raw.append("quiet")
        elif late >= 0.40 * early:
            raw.append("sustained")
        elif mid >= 0.50 * early:
            raw.append("fades later")
        else:
            raw.append("front-loaded")
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for label in raw:
        totals[label] = totals.get(label, 0) + 1
    names = []
    for label, clicks in zip(raw, mean_clicks):
        counts[label] = counts.get(label, 0) + 1
        name = label if totals[label] == 1 else f"{label} {counts[label]}"
        names.append(f"{name} ({clicks:.0f} clicks)")
    return names


def plot_engagement_clusters(ctx: Context, path):
    """Retrospective 12-week profiles. Not an early-warning feature."""
    import matplotlib.pyplot as plt

    students = _started(ctx.cohort)[KEYS + ["final_result"]]
    daily = ctx.daily.loc[(ctx.daily["date"] >= 0) & (ctx.daily["date"] < 84), KEYS + ["date", "clicks"]].copy()
    daily["week"] = (daily["date"] // 7).astype(int)
    weekly = daily.groupby(KEYS + ["week"], sort=False)["clicks"].sum().reset_index()
    matrix = weekly.pivot_table(index=KEYS, columns="week", values="clicks", fill_value=0, aggfunc="sum")
    matrix = matrix.reindex(columns=range(12), fill_value=0)
    matrix = students.merge(matrix.reset_index(), on=KEYS, how="left")
    week_cols = list(range(12))
    for column in week_cols:
        if column not in matrix.columns:
            matrix[column] = 0
        matrix[column] = matrix[column].fillna(0)
    totals = matrix[week_cols].sum(axis=1)
    active = matrix.loc[totals > 0].copy()
    inactive = matrix.loc[totals <= 0]
    values = np.log1p(active[week_cols].to_numpy(dtype=float))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    scaled = values / np.clip(norms, 1e-9, None)
    model = KMeans(n_clusters=4, n_init=10, random_state=0)
    labels = model.fit_predict(scaled)
    active = active.copy()
    active["cluster"] = labels
    active["_total"] = active[week_cols].sum(axis=1)
    mean_clicks = (
        active.groupby("cluster")["_total"].mean().reindex(range(4)).fillna(0).to_numpy()
    )
    profiles = np.vstack(
        [
            active.loc[active["cluster"] == cluster, week_cols].mean().to_numpy()
            for cluster in range(4)
        ]
    )
    names = _cluster_labels(profiles, mean_clicks)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for cluster, color, name in zip(range(4), colors, names):
        axes[0].plot(range(12), profiles[cluster], marker="o", color=color, label=name)
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Mean clicks")
    axes[0].set_title("Retrospective click profiles")
    axes[0].legend(fontsize=8)
    pieces = [active]
    if len(inactive):
        quiet = inactive.copy()
        quiet["cluster"] = -1
        pieces.append(quiet)
    combined = pd.concat(pieces, ignore_index=True)
    order = list(range(4)) + ([-1] if (combined["cluster"] == -1).any() else [])
    tick_labels = names + (["no clicks"] if -1 in order else [])
    shares = []
    for cluster in order:
        part = combined.loc[combined["cluster"] == cluster, "final_result"]
        counts = part.value_counts(normalize=True)
        shares.append([float(counts.get(outcome, 0.0)) for outcome in OUTCOME_ORDER])
    shares = np.asarray(shares)
    left = np.zeros(len(order))
    y = np.arange(len(order))
    for index, outcome in enumerate(OUTCOME_ORDER):
        axes[1].barh(y, shares[:, index], left=left, color=OUTCOME_COLORS[outcome], label=outcome)
        left = left + shares[:, index]
    axes[1].set_yticks(y, tick_labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Outcome mix of each profile")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    figure.suptitle("Full first-12-week patterns, not used as model features", fontsize=12)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.22)
    save(figure, path)
    return figure


def make_eda_figures(ctx: Context, figure_dir) -> dict[str, str]:
    figure_dir = figure_dir
    paths = {
        "outcome_mix": figure_dir / "outcome_mix.png",
        "weekly_clicks": figure_dir / "weekly_clicks.png",
        "first_tma": figure_dir / "first_tma.png",
        "demographics": figure_dir / "demographics.png",
        "withdrawal_km": figure_dir / "withdrawal_km.png",
        "engagement_clusters": figure_dir / "engagement_clusters.png",
    }
    plot_outcome_mix(ctx.cohort, paths["outcome_mix"])
    plot_weekly_clicks(ctx, paths["weekly_clicks"])
    plot_first_tma(ctx, paths["first_tma"])
    plot_demographics(ctx.cohort, paths["demographics"])
    plot_withdrawal_km(ctx.cohort, paths["withdrawal_km"])
    plot_engagement_clusters(ctx, paths["engagement_clusters"])
    return {name: str(path) for name, path in paths.items()}
