"""Cutoff-safe feature tables.

Every time-dependent column is built inside ``build_features``. Click totals are
cumulative sums taken on the last day at or before the cutoff, and assessment
scores enter only when ``date_submitted`` is at or before the cutoff. Exams are
omitted: they are the end-of-module result and are missing for most students.

``date_unregistration`` is an eligibility rule, not a feature. Students with a
negative unregistration date never started and are dropped at every cutoff.
Students who have already unregistered by the cutoff are dropped from the main
model, because a tutor already knows they have left. ``active_only=False``
keeps those students so the inflation can be measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oulad.config import (
    ASSESSMENT_FEATURES,
    AT_RISK_RESULTS,
    CATEGORICAL_FEATURES,
    CUTOFFS,
    IMD_HIGH,
    IMD_LOW,
    IMD_MID,
    KEYS,
    NON_FEATURES,
    PROTECTED_FEATURES,
    PROXY_FEATURES,
    STATIC_FEATURES,
)

LEAKAGE_RULES = (
    "Clicks after the cutoff are excluded. Totals are the cumulative sum through the last observed day at or before the cutoff.",
    "The last-7-day window is the seven days ending on the cutoff: dates in (cutoff-7, cutoff].",
    "Assessment scores count only when date_submitted is at or before the cutoff. Exam rows are excluded.",
    "An assessment is due only when its scheduled date is at or before the cutoff. That calendar is known in advance.",
    "date_unregistration is never a feature. It only decides who is still enrolled.",
    "Students with date_unregistration < 0 are excluded at every cutoff. They never started.",
    "The main model also excludes students with date_unregistration <= cutoff. A sensitivity fit keeps them.",
    "final_result is the label only. id_student and code_presentation are not features.",
    "Banked scores (is_banked = 1) are treated as known from day 0, so a late transfer date cannot hide or delay them.",
)


def imd_group_of(value) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>", "Missing"}:
        return "Missing"
    if not text.endswith("%"):
        text = text + "%"
    if text in IMD_LOW:
        return "low"
    if text in IMD_MID:
        return "mid"
    if text in IMD_HIGH:
        return "high"
    return "Missing"


def activity_column(activity_type: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(activity_type).strip().lower()).strip("_")
    if not slug:
        slug = "unknown"
    return f"clicks_type_{slug}"


def eligible_mask(cohort: pd.DataFrame, cutoff: int, active_only: bool) -> pd.Series:
    unreg = cohort["date_unregistration"]
    never_started = unreg.notna() & (unreg < 0)
    if not active_only:
        return ~never_started
    already_left = unreg.notna() & (unreg <= cutoff)
    return ~never_started & ~already_left


@dataclass
class Context:
    cohort: pd.DataFrame
    daily: pd.DataFrame
    typed: pd.DataFrame
    assessments: pd.DataFrame
    submissions: pd.DataFrame
    prestart: pd.DataFrame
    activity_types: list[str]


def _fill_category(series: pd.Series) -> pd.Series:
    out = series.astype("object")
    missing = out.isna() | out.isin(["", "nan", "None", "<NA>"])
    out = out.where(~missing, "Missing")
    return out


def prepare(tables: dict[str, pd.DataFrame]) -> Context:
    """Join the cohort and reduce the click log to daily running totals."""
    info = tables["student_info"].copy()
    registration = tables["student_registration"][
        KEYS + ["date_registration", "date_unregistration"]
    ].copy()
    if info.duplicated(KEYS).any():
        raise ValueError("student_info has duplicate module-presentation-student keys.")
    if registration.duplicated(KEYS).any():
        raise ValueError("student_registration has duplicate keys.")
    cohort = info.merge(registration, on=KEYS, how="inner", validate="one_to_one")
    courses = tables["courses"][["code_module", "code_presentation", "module_length"]]
    cohort = cohort.merge(courses, on=["code_module", "code_presentation"], how="left")
    if cohort["module_length"].isna().any():
        missing = int(cohort["module_length"].isna().sum())
        print(f"Warning: filled module length for {missing} rows with 270.", flush=True)
        cohort["module_length"] = cohort["module_length"].fillna(270)
    cohort["at_risk"] = cohort["final_result"].isin(AT_RISK_RESULTS).astype("int8")
    # map() can skip pd.NA. Build the group from the raw values instead.
    cohort["imd_group"] = [imd_group_of(value) for value in cohort["imd_band"].tolist()]
    for column in (
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ):
        cohort[column] = _fill_category(cohort[column])
    cohort["num_of_prev_attempts"] = pd.to_numeric(cohort["num_of_prev_attempts"], errors="coerce")
    cohort["studied_credits"] = pd.to_numeric(cohort["studied_credits"], errors="coerce")
    cohort["date_registration"] = pd.to_numeric(cohort["date_registration"], errors="coerce")
    cohort["date_unregistration"] = pd.to_numeric(cohort["date_unregistration"], errors="coerce")

    assessments = tables["assessments"].copy()
    assessments["assessment_type"] = _fill_category(assessments["assessment_type"])
    submissions = tables["student_assessment"].merge(
        assessments[
            [
                "id_assessment",
                "code_module",
                "code_presentation",
                "assessment_type",
                "date_due",
                "weight",
            ]
        ],
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )
    dropped = len(tables["student_assessment"]) - len(submissions)
    if dropped:
        print(f"Dropped {dropped} submission rows with an unknown assessment id.", flush=True)
    banked = submissions["is_banked"] == 1
    # A banked score was earned in an earlier presentation. Clip the date so it
    # is available from day 0 and cannot appear as a future submission.
    submissions.loc[banked, "date_submitted"] = (
        submissions.loc[banked, "date_submitted"].fillna(0).clip(upper=0)
    )
    n_banked = int(banked.sum())
    if n_banked:
        print(f"Clipped {n_banked} banked submissions to day 0.", flush=True)

    vle = tables["vle"][["id_site", "code_module", "code_presentation", "activity_type"]].copy()
    vle["activity_type"] = _fill_category(vle["activity_type"])
    if vle["id_site"].is_unique:
        site_type = vle[["id_site", "activity_type"]]
        merged = tables["student_vle"].merge(site_type, on="id_site", how="left")
    else:
        merged = tables["student_vle"].merge(
            vle, on=["id_site", "code_module", "code_presentation"], how="left"
        )
    unknown = int(merged["activity_type"].isna().sum())
    if unknown:
        print(f"Warning: {unknown} click rows have no activity type.", flush=True)
        merged["activity_type"] = merged["activity_type"].fillna("unknown")
    typed = (
        merged.groupby(KEYS + ["activity_type", "date"], sort=False)["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "clicks"})
    )
    del merged
    typed = typed.sort_values(KEYS + ["activity_type", "date"], kind="mergesort")
    typed["cum_clicks"] = typed.groupby(KEYS + ["activity_type"], sort=False)["clicks"].cumsum()
    if typed.duplicated(KEYS + ["activity_type", "date"]).any():
        raise AssertionError("Typed daily clicks are not unique.")

    daily = (
        typed.groupby(KEYS + ["date"], sort=False)["clicks"].sum().reset_index()
    )
    daily = daily.sort_values(KEYS + ["date"], kind="mergesort")
    daily["cum_clicks"] = daily.groupby(KEYS, sort=False)["clicks"].cumsum()
    daily["cum_days"] = daily.groupby(KEYS, sort=False).cumcount() + 1
    if daily.duplicated(KEYS + ["date"]).any():
        raise AssertionError("Daily clicks are not unique.")

    prestart = (
        daily.loc[daily["date"] < 0, KEYS + ["clicks"]]
        .groupby(KEYS, sort=False)["clicks"]
        .sum()
        .rename("clicks_prestart")
        .reset_index()
    )
    activity_types = sorted(vle["activity_type"].dropna().unique().tolist())
    slugs = [activity_column(activity) for activity in activity_types]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"Activity type names collide after slugging: {slugs}")
    print(
        f"Cohort {len(cohort):,} rows; daily click rows {len(daily):,}; "
        f"activity types {len(activity_types)}.",
        flush=True,
    )
    return Context(
        cohort=cohort,
        daily=daily,
        typed=typed,
        assessments=assessments,
        submissions=submissions,
        prestart=prestart,
        activity_types=activity_types,
    )


def exclusion_report(cohort: pd.DataFrame) -> dict:
    unreg = cohort["date_unregistration"]
    never = unreg.notna() & (unreg < 0)
    report = {
        "n_rows": int(len(cohort)),
        "n_students": int(cohort["id_student"].nunique()),
        "n_never_started": int(never.sum()),
        "n_unregistered_on_day_0": int((unreg == 0).sum()),
        "n_withdrawn_label_but_no_date": int(
            ((cohort["final_result"] == "Withdrawn") & unreg.isna()).sum()
        ),
        "final_result_counts": {
            str(key): int(value)
            for key, value in cohort["final_result"].value_counts().sort_index().items()
        },
    }
    for cutoff in CUTOFFS:
        eligible = eligible_mask(cohort, cutoff, active_only=True)
        already = (~never) & unreg.notna() & (unreg <= cutoff)
        report[f"n_eligible_day_{cutoff}"] = int(eligible.sum())
        report[f"n_already_left_day_{cutoff}"] = int(already.sum())
        report[f"base_rate_day_{cutoff}"] = float(cohort.loc[eligible, "at_risk"].mean())
    return report


def _snapshot(frame: pd.DataFrame, key_columns: list[str], cutoff: int) -> pd.DataFrame:
    """Last row with date <= cutoff. ``frame`` must be sorted by keys then date."""
    subset = frame.loc[frame["date"] <= cutoff]
    if subset.empty:
        return subset.iloc[0:0]
    return subset.drop_duplicates(key_columns, keep="last")


def feature_columns(frame: pd.DataFrame, drop: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Approved model columns. Anything time-varying has already been cut off."""
    banned = set(NON_FEATURES) | set(drop)
    present = [column for column in frame.columns if column not in banned]

    def sort_key(column: str) -> tuple[int, str]:
        if column in STATIC_FEATURES:
            return (0, column)
        if column.startswith("clicks_") or column in {
            "active_days",
            "days_since_last_click",
            "never_active",
        }:
            return (1, column)
        if column in ASSESSMENT_FEATURES:
            return (2, column)
        return (3, column)

    return sorted(present, key=sort_key)


def assert_safe_columns(columns: list[str]) -> None:
    leaked = [column for column in columns if column in NON_FEATURES]
    if leaked:
        raise AssertionError(f"Feature list contains non-features: {leaked}")


def build_features(ctx: Context, cutoff: int, active_only: bool = True) -> pd.DataFrame:
    """One row per student still in scope at ``cutoff``, using only earlier data."""
    mask = eligible_mask(ctx.cohort, cutoff, active_only=active_only)
    base_columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
        "num_of_prev_attempts",
        "studied_credits",
        "date_registration",
        "final_result",
        "at_risk",
        "date_unregistration",
        "imd_group",
    ]
    base = ctx.cohort.loc[mask, base_columns].copy()

    snap = _snapshot(ctx.daily, KEYS, cutoff)
    snap7 = _snapshot(ctx.daily, KEYS, cutoff - 7)
    if not snap.empty:
        snap = snap.rename(
            columns={
                "date": "last_click_date",
                "cum_clicks": "clicks_total",
                "cum_days": "active_days",
            }
        )[KEYS + ["last_click_date", "clicks_total", "active_days"]]
        base = base.merge(snap, on=KEYS, how="left")
    else:
        base["last_click_date"] = np.nan
        base["clicks_total"] = np.nan
        base["active_days"] = np.nan
    if not snap7.empty:
        previous = snap7[KEYS + ["cum_clicks"]].rename(columns={"cum_clicks": "_clicks_through_prev"})
        base = base.merge(previous, on=KEYS, how="left")
    else:
        base["_clicks_through_prev"] = np.nan
    base["clicks_last_7d"] = base["clicks_total"] - base["_clicks_through_prev"].fillna(0)
    base.loc[base["clicks_total"].isna(), "clicks_last_7d"] = 0
    base["never_active"] = base["last_click_date"].isna().astype("int8")
    base["days_since_last_click"] = cutoff - base["last_click_date"]
    base["clicks_total"] = base["clicks_total"].fillna(0)
    base["active_days"] = base["active_days"].fillna(0).astype("int32")
    base["clicks_last_7d"] = base["clicks_last_7d"].fillna(0)

    base = base.merge(ctx.prestart, on=KEYS, how="left")
    base["clicks_prestart"] = base["clicks_prestart"].fillna(0)

    typed_snap = _snapshot(ctx.typed, KEYS + ["activity_type"], cutoff)
    if typed_snap.empty:
        wide = pd.DataFrame(columns=KEYS)
    else:
        wide = (
            typed_snap.set_index(KEYS + ["activity_type"])["cum_clicks"]
            .unstack("activity_type")
            .reset_index()
        )
        rename = {
            column: activity_column(column)
            for column in wide.columns
            if column not in KEYS
        }
        wide = wide.rename(columns=rename)
    for activity in ctx.activity_types:
        column = activity_column(activity)
        if column not in wide.columns:
            wide[column] = 0
    type_columns = [activity_column(activity) for activity in ctx.activity_types]
    if type_columns:
        base = base.merge(wide[KEYS + type_columns], on=KEYS, how="left")
        base[type_columns] = base[type_columns].fillna(0)

    scheduled = ctx.assessments.loc[
        ctx.assessments["assessment_type"].str.lower() != "exam"
    ].copy()
    due = scheduled.loc[scheduled["date_due"].notna() & (scheduled["date_due"] <= cutoff)]
    n_due = (
        due.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    )
    base = base.merge(n_due, on=["code_module", "code_presentation"], how="left")
    base["n_due"] = base["n_due"].fillna(0).astype("int16")

    taken = ctx.submissions.loc[
        ctx.submissions["assessment_type"].str.lower() != "exam"
    ].copy()
    taken = taken.loc[taken["date_submitted"].notna() & (taken["date_submitted"] <= cutoff)]
    due_ids = set(due["id_assessment"].tolist())
    taken_due = taken.loc[taken["id_assessment"].isin(due_ids)]
    n_submitted = (
        taken_due.groupby(KEYS)["id_assessment"].nunique().rename("n_submitted_due").reset_index()
    )
    base = base.merge(n_submitted, on=KEYS, how="left")
    base["n_submitted_due"] = base["n_submitted_due"].fillna(0).astype("int16")
    base["submit_rate"] = np.where(
        base["n_due"] > 0, base["n_submitted_due"] / base["n_due"], np.nan
    )

    scored = taken.loc[taken["score"].notna()]
    if scored.empty:
        base["asmt_mean_score"] = np.nan
        base["asmt_min_score"] = np.nan
    else:
        aggregated = (
            scored.groupby(KEYS)["score"]
            .agg(asmt_mean_score="mean", asmt_min_score="min")
            .reset_index()
        )
        base = base.merge(aggregated, on=KEYS, how="left")

    tmas = scheduled.loc[
        (scheduled["assessment_type"].str.lower() == "tma") & scheduled["date_due"].notna()
    ]
    tmas = tmas.sort_values(
        ["code_module", "code_presentation", "date_due", "id_assessment"], kind="mergesort"
    )
    first = tmas.groupby(["code_module", "code_presentation"], sort=False).head(1)
    first = first[["code_module", "code_presentation", "id_assessment", "date_due"]].rename(
        columns={"id_assessment": "tma_id", "date_due": "tma_date"}
    )
    base = base.merge(first, on=["code_module", "code_presentation"], how="left")
    first_ids = set(first["tma_id"].tolist())
    tma_rows = ctx.submissions.loc[ctx.submissions["id_assessment"].isin(first_ids)].copy()
    tma_rows = tma_rows.sort_values("date_submitted", kind="mergesort")
    tma_rows = tma_rows.drop_duplicates(KEYS, keep="first")
    tma_rows = tma_rows[KEYS + ["date_submitted", "score"]].rename(
        columns={"date_submitted": "tma_submitted", "score": "tma_score"}
    )
    base = base.merge(tma_rows, on=KEYS, how="left")

    due_now = base["tma_date"].notna() & (base["tma_date"] <= cutoff)
    submitted_now = base["tma_submitted"].notna() & (base["tma_submitted"] <= cutoff)
    base["first_tma_due"] = due_now.astype("int8")
    base["first_tma_missing"] = (due_now & ~submitted_now).astype("int8")
    base["first_tma_days_late"] = np.nan
    base.loc[submitted_now, "first_tma_days_late"] = (
        base.loc[submitted_now, "tma_submitted"] - base.loc[submitted_now, "tma_date"]
    )
    base["first_tma_score"] = np.nan
    base.loc[submitted_now, "first_tma_score"] = base.loc[submitted_now, "tma_score"]

    base["cutoff"] = cutoff
    helper_columns = [
        "last_click_date",
        "_clicks_through_prev",
        "tma_id",
        "tma_date",
        "tma_submitted",
        "tma_score",
    ]
    base = base.drop(columns=helper_columns)
    if base.duplicated(KEYS).any():
        raise AssertionError("Feature table has duplicate student keys.")
    unreg = base["date_unregistration"]
    if (unreg.notna() & (unreg < 0)).any():
        raise AssertionError("Never-started students entered the feature table.")
    if active_only and (unreg.notna() & (unreg <= cutoff)).any():
        raise AssertionError("Already-unregistered students entered an active-only table.")
    assert_safe_columns(feature_columns(base))
    return base.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def family_of(column: str) -> str:
    if column in PROTECTED_FEATURES:
        return "protected"
    if column in PROXY_FEATURES:
        return "proxy"
    if column.startswith("clicks_") or column in {
        "active_days",
        "days_since_last_click",
        "never_active",
    }:
        return "engagement"
    if column in ASSESSMENT_FEATURES:
        return "assessment"
    if column in CATEGORICAL_FEATURES or column in STATIC_FEATURES:
        return "other static"
    return "other static"
