"""Train/test splits.

Temporal: fit on 2013 presentations, test on 2014. This is the split to quote.
Random: a single stratified row split. It is the optimistic comparison, because
the same presentation and sometimes the same student land on both sides.
Leave-one-module-out: module transfer. It is not also time-shifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from oulad.config import RANDOM_SEED, TEST_PRESENTATIONS, TRAIN_PRESENTATIONS


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["code_presentation"].isin(TRAIN_PRESENTATIONS)].copy()
    test = frame.loc[frame["code_presentation"].isin(TEST_PRESENTATIONS)].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty side.")
    return train, test


def inner_temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2013B to fit, 2013J to choose the small LightGBM grid. 2014 is untouched."""
    train = frame.loc[frame["code_presentation"] == "2013B"].copy()
    validation = frame.loc[frame["code_presentation"] == "2013J"].copy()
    if train.empty or validation.empty:
        raise ValueError("Inner 2013B/2013J split is empty. Check the cutoff cohort.")
    return train, validation


def iter_leave_one_module_out(frame: pd.DataFrame):
    modules = sorted(frame["code_module"].dropna().unique())
    for module in modules:
        test = frame.loc[frame["code_module"] == module].copy()
        train = frame.loc[frame["code_module"] != module].copy()
        yield str(module), train, test


def random_split(
    frame: pd.DataFrame, seed: int = RANDOM_SEED, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One stratified row split. Rows are not grouped by student or presentation."""
    working = frame.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        working.index,
        test_size=test_size,
        random_state=seed,
        stratify=working["at_risk"].to_numpy(),
    )
    return working.loc[train_idx].copy(), working.loc[test_idx].copy()


def student_overlap(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Share of test rows whose student id also appears in train."""
    seen = set(train["id_student"].tolist())
    return float(test["id_student"].isin(seen).mean())


def presentation_overlap(train: pd.DataFrame, test: pd.DataFrame) -> float:
    train_keys = set(zip(train["code_module"], train["code_presentation"]))
    pairs = list(zip(test["code_module"], test["code_presentation"]))
    if not pairs:
        return float("nan")
    return float(np.mean([pair in train_keys for pair in pairs]))
