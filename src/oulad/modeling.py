"""Logistic regression baseline and a small-grid LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from oulad.config import CATEGORICAL_FEATURES, RANDOM_SEED
from oulad.features import feature_columns

DEFAULT_LGBM = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "force_row_wise": True,
}

LGBM_GRID = (
    {"num_leaves": 31, "learning_rate": 0.05},
    {"num_leaves": 63, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.1},
)


def drop_constants(train: pd.DataFrame, columns: list[str]) -> list[str]:
    kept = []
    for column in columns:
        if train[column].nunique(dropna=True) > 1:
            kept.append(column)
    if not kept:
        raise ValueError("Every candidate feature is constant on the training fold.")
    return kept


def _split_types(columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
    numeric = [column for column in columns if column not in CATEGORICAL_FEATURES]
    return numeric, categorical


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    for column in columns:
        if column.startswith("clicks_"):
            numeric[column] = np.log1p(numeric[column].clip(lower=0))
    numeric = numeric.reset_index(drop=True)
    numeric.columns = list(columns)
    return numeric


def _fill_categories(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    filled = frame[columns].astype("object")
    return filled.where(filled.notna(), "Missing").reset_index(drop=True)


def _positive_probability(model, matrix) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Model classes do not include the positive label: {classes}")
    return probabilities[:, classes.index(1)]


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    columns = drop_constants(train, columns)
    numeric_cols, categorical_cols = _split_types(columns)
    train_num = _numeric_frame(train, numeric_cols) if numeric_cols else pd.DataFrame(index=range(len(train)))
    test_num = _numeric_frame(test, numeric_cols) if numeric_cols else pd.DataFrame(index=range(len(test)))
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_num = scaler.fit_transform(imputer.fit_transform(train_num))
        test_num = scaler.transform(imputer.transform(test_num))
    else:
        train_num = np.zeros((len(train), 0))
        test_num = np.zeros((len(test), 0))
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_cat = encoder.fit_transform(_fill_categories(train, categorical_cols))
        test_cat = encoder.transform(_fill_categories(test, categorical_cols))
        train_x = np.hstack([train_num, train_cat])
        test_x = np.hstack([test_num, test_cat])
    else:
        train_x = np.asarray(train_num)
        test_x = np.asarray(test_num)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train["at_risk"].to_numpy().astype(int))
    return _positive_probability(model, test_x)


def _design_lgbm(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = drop_constants(train, columns)
    numeric_cols, categorical_cols = _split_types(columns)
    pieces_train = []
    pieces_test = []
    if numeric_cols:
        pieces_train.append(_numeric_frame(train, numeric_cols))
        pieces_test.append(_numeric_frame(test, numeric_cols))
    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        train_raw = _fill_categories(train, categorical_cols)
        test_raw = _fill_categories(test, categorical_cols)
        train_codes = np.asarray(encoder.fit_transform(train_raw), dtype=np.int32)
        test_codes = np.asarray(encoder.transform(test_raw), dtype=np.int32)
        pieces_train.append(pd.DataFrame(train_codes, columns=categorical_cols))
        pieces_test.append(pd.DataFrame(test_codes, columns=categorical_cols))
    train_x = pd.concat(pieces_train, axis=1)
    test_x = pd.concat(pieces_test, axis=1)
    train_x.columns = list(train_x.columns)
    test_x.columns = list(test_x.columns)
    return train_x, test_x, categorical_cols


def _lgbm(params: dict, y_train: np.ndarray) -> LGBMClassifier:
    merged = {**DEFAULT_LGBM, **params}
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    merged["scale_pos_weight"] = negatives / positives
    return LGBMClassifier(**merged)


def fit_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    params: dict | None = None,
    return_details: bool = False,
):
    """Return P(at_risk) for ``test``. Optionally return the booster and matrix."""
    if train["at_risk"].nunique() < 2:
        raise ValueError("Training fold has a single class.")
    train_x, test_x, categorical_cols = _design_lgbm(train, test, columns)
    y_train = train["at_risk"].to_numpy().astype(int)
    model = _lgbm(params or {}, y_train)
    fit_kwargs = {}
    if categorical_cols:
        fit_kwargs["categorical_feature"] = categorical_cols
    model.fit(train_x, y_train, **fit_kwargs)
    scores = _positive_probability(model, test_x)
    if not return_details:
        return scores
    return scores, {"model": model, "X_test": test_x, "features": list(train_x.columns)}


def select_hyperparams(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> dict:
    """Pick num_leaves and learning_rate on 2013J. Tree count comes from early stopping."""
    best = None
    for grid in LGBM_GRID:
        if train["at_risk"].nunique() < 2 or validation["at_risk"].nunique() < 2:
            raise ValueError("Hyperparameter split is missing a class.")
        train_x, valid_x, categorical_cols = _design_lgbm(train, validation, columns)
        y_train = train["at_risk"].to_numpy().astype(int)
        y_valid = validation["at_risk"].to_numpy().astype(int)
        model = _lgbm({**grid, "n_estimators": 400}, y_train)
        fit_kwargs = {
            "eval_X": valid_x,
            "eval_y": y_valid,
            "eval_metric": "average_precision",
            "callbacks": [early_stopping(40, verbose=False), log_evaluation(period=0)],
        }
        if categorical_cols:
            fit_kwargs["categorical_feature"] = categorical_cols
        model.fit(train_x, y_train, **fit_kwargs)
        scores = _positive_probability(model, valid_x)
        score = float(average_precision_score(y_valid, scores))
        n_trees = int(model.best_iteration_ or 0)
        if n_trees <= 0:
            n_trees = 200
        n_trees = max(n_trees, 50)
        print(
            f"  grid leaves={grid['num_leaves']} lr={grid['learning_rate']}: "
            f"val PR-AUC {score:.4f}, trees {n_trees}",
            flush=True,
        )
        if best is None or score > best["val_pr_auc"]:
            best = {
                "num_leaves": grid["num_leaves"],
                "learning_rate": grid["learning_rate"],
                "n_estimators": n_trees,
                "val_pr_auc": score,
            }
    print(
        f"Selected leaves={best['num_leaves']} lr={best['learning_rate']} "
        f"trees={best['n_estimators']} (2013J PR-AUC {best['val_pr_auc']:.4f}).",
        flush=True,
    )
    return best


def gain_importance(model: LGBMClassifier, feature_names: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    names = list(feature_names)
    if len(gain) != len(names):
        names = list(model.booster_.feature_name())
    total = float(np.sum(gain)) or 1.0
    frame = pd.DataFrame({"feature": names, "gain": gain})
    frame["share"] = frame["gain"] / total
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def write_shap(model, matrix: pd.DataFrame, path, sample_size: int = 2000) -> None:
    import matplotlib.pyplot as plt
    import shap

    from oulad.plotting import save

    sample = matrix.sample(n=min(sample_size, len(matrix)), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    if hasattr(values, "values"):
        shap.plots.beeswarm(values, max_display=15, show=False)
    else:
        shap.summary_plot(values, sample, max_display=15, show=False)
    figure = plt.gcf()
    figure.suptitle("Day 28 temporal model, what moves the risk score", fontsize=13, y=1.02)
    save(figure, path)
    plt.close(figure)


def modeling_columns(frame: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[str]:
    return feature_columns(frame, drop=drop)
