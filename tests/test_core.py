"""Leakage, split, metric, and model-API checks that do not need the full extract."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oulad.fairness import check_against_fairlearn, thresholds_for_recall, apply_group_thresholds
from oulad.features import build_features, feature_columns, prepare
from oulad.load import normalize_tables
from oulad.metrics import flags_at_budget, fnr_gap, group_table, recall_from_flags, within_module_flags
from oulad.modeling import fit_lgbm, fit_logistic, select_hyperparams
from oulad.splits import random_split, temporal_split


def _tables() -> dict[str, pd.DataFrame]:
    courses = pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "length": 200},
            {"code_module": "AAA", "code_presentation": "2014J", "length": 200},
        ]
    )
    assessments = pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "id_assessment": 1, "assessment_type": "TMA", "date": 20, "weight": 10},
            {"code_module": "AAA", "code_presentation": "2013J", "id_assessment": 2, "assessment_type": "TMA", "date": 50, "weight": 10},
            {"code_module": "AAA", "code_presentation": "2013J", "id_assessment": 3, "assessment_type": "CMA", "date": 30, "weight": 5},
            {"code_module": "AAA", "code_presentation": "2013J", "id_assessment": 4, "assessment_type": "Exam", "date": None, "weight": 100},
            {"code_module": "AAA", "code_presentation": "2014J", "id_assessment": 5, "assessment_type": "TMA", "date": 20, "weight": 10},
            {"code_module": "AAA", "code_presentation": "2014J", "id_assessment": 6, "assessment_type": "Exam", "date": None, "weight": 100},
        ]
    )
    activities = ["forumng", "quiz", "url", "homepage"]
    vle_rows = []
    site = 10
    for presentation in ("2013J", "2014J"):
        for activity in activities:
            vle_rows.append(
                {
                    "id_site": site,
                    "code_module": "AAA",
                    "code_presentation": presentation,
                    "activity_type": activity,
                    "week_from": None,
                    "week_to": None,
                }
            )
            site += 1
    vle = pd.DataFrame(vle_rows)
    info_rows = []
    demographics = {
        1: ("M", "0-10%", "Pass"),
        2: ("F", "10-20%", "Withdrawn"),
        3: ("M", "20-30%", "Withdrawn"),
        4: ("F", "90-100%", "Fail"),
        5: ("M", "50-60%", "Distinction"),
    }
    registration = {
        1: (-20, None, "2013J"),
        2: (-15, 10, "2013J"),
        3: (-12, -3, "2013J"),
        4: (-8, None, "2013J"),
        5: (-5, None, "2014J"),
    }
    info = []
    regs = []
    for student, (gender, band, result) in demographics.items():
        registered, unregistered, presentation = registration[student]
        info.append(
            {
                "code_module": "AAA",
                "code_presentation": presentation,
                "id_student": student,
                "gender": gender,
                "region": "London",
                "highest_education": "A Level or Equivalent",
                "imd_band": band,
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": result,
            }
        )
        regs.append(
            {
                "code_module": "AAA",
                "code_presentation": presentation,
                "id_student": student,
                "date_registration": registered,
                "date_unregistration": unregistered,
            }
        )
    # Sites: 10 forum, 11 quiz, 12 url, 13 homepage for 2013J. 14+ for 2014J.
    clicks = pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": -5, "sum_click": 3},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": 10, "sum_click": 5},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": 21, "sum_click": 6},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": 22, "sum_click": 4},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 12, "date": 28, "sum_click": 1},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 12, "date": 29, "sum_click": 7},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 11, "date": 40, "sum_click": 100},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 10, "date": 2, "sum_click": 4},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 10, "date": 30, "sum_click": 50},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 3, "id_site": 10, "date": 1, "sum_click": 9},
            {"code_module": "AAA", "code_presentation": "2014J", "id_student": 5, "id_site": 15, "date": 5, "sum_click": 2},
        ]
    )
    submissions = pd.DataFrame(
        [
            {"id_assessment": 1, "id_student": 1, "date_submitted": 22, "is_banked": 0, "score": 80},
            {"id_assessment": 2, "id_student": 1, "date_submitted": 55, "is_banked": 0, "score": 10},
            {"id_assessment": 4, "id_student": 1, "date_submitted": 1, "is_banked": 0, "score": 0},
            {"id_assessment": 5, "id_student": 5, "date_submitted": 18, "is_banked": 0, "score": 90},
        ]
    )
    return {
        "courses": courses,
        "assessments": assessments,
        "vle": vle,
        "student_info": pd.DataFrame(info),
        "student_registration": pd.DataFrame(regs),
        "student_assessment": submissions,
        "student_vle": clicks,
    }


def _row(frame: pd.DataFrame, student: int) -> pd.Series:
    rows = frame.loc[frame["id_student"] == student]
    if len(rows) != 1:
        raise AssertionError(f"student {student} has {len(rows)} rows")
    return rows.iloc[0]


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = prepare(normalize_tables(_tables()))

    def test_cutoff_28_excludes_future_and_people_who_left(self):
        frame = build_features(self.ctx, 28, active_only=True)
        self.assertEqual(set(frame["id_student"]), {1, 4, 5})
        student = _row(frame, 1)
        self.assertAlmostEqual(student["clicks_total"], 19)
        self.assertAlmostEqual(student["clicks_type_forumng"], 18)
        self.assertAlmostEqual(student["clicks_type_url"], 1)
        self.assertAlmostEqual(student["clicks_type_quiz"], 0)
        self.assertAlmostEqual(student["clicks_type_homepage"], 0)
        self.assertAlmostEqual(student["clicks_prestart"], 3)
        self.assertAlmostEqual(student["clicks_last_7d"], 5)
        self.assertEqual(student["active_days"], 5)
        self.assertAlmostEqual(student["days_since_last_click"], 0)
        self.assertEqual(student["never_active"], 0)
        self.assertAlmostEqual(student["asmt_mean_score"], 80)
        self.assertAlmostEqual(student["asmt_min_score"], 80)
        self.assertEqual(student["n_due"], 1)
        self.assertEqual(student["n_submitted_due"], 1)
        self.assertAlmostEqual(student["submit_rate"], 1)
        self.assertEqual(student["first_tma_due"], 1)
        self.assertEqual(student["first_tma_missing"], 0)
        self.assertAlmostEqual(student["first_tma_days_late"], 2)
        self.assertAlmostEqual(student["first_tma_score"], 80)
        self.assertAlmostEqual(student["date_registration"], -20)
        self.assertEqual(student["imd_group"], "low")
        self.assertEqual(student["at_risk"], 0)

        quiet = _row(frame, 4)
        self.assertEqual(quiet["never_active"], 1)
        self.assertTrue(pd.isna(quiet["days_since_last_click"]))
        self.assertAlmostEqual(quiet["clicks_total"], 0)
        self.assertEqual(quiet["first_tma_missing"], 1)
        self.assertAlmostEqual(quiet["submit_rate"], 0)
        self.assertTrue(pd.isna(quiet["asmt_mean_score"]))
        self.assertEqual(quiet["at_risk"], 1)

        later = _row(frame, 5)
        self.assertAlmostEqual(later["clicks_total"], 2)
        self.assertAlmostEqual(later["clicks_last_7d"], 0)
        self.assertAlmostEqual(later["days_since_last_click"], 23)
        self.assertAlmostEqual(later["first_tma_days_late"], -2)
        self.assertAlmostEqual(later["first_tma_score"], 90)
        self.assertEqual(later["at_risk"], 0)

    def test_cutoff_0_has_no_assessment_and_keeps_not_yet_withdrawn(self):
        frame = build_features(self.ctx, 0, active_only=True)
        self.assertEqual(set(frame["id_student"]), {1, 2, 4, 5})
        student = _row(frame, 1)
        self.assertAlmostEqual(student["clicks_total"], 3)
        self.assertAlmostEqual(student["clicks_last_7d"], 3)
        self.assertAlmostEqual(student["days_since_last_click"], 5)
        self.assertEqual(student["n_due"], 0)
        self.assertTrue(pd.isna(student["submit_rate"]))
        self.assertTrue(pd.isna(student["asmt_mean_score"]))
        self.assertEqual(student["first_tma_due"], 0)
        self.assertEqual(student["first_tma_missing"], 0)
        self.assertEqual(_row(frame, 2)["never_active"], 1)

    def test_keeping_withdrawn_students_still_hides_future_clicks(self):
        frame = build_features(self.ctx, 28, active_only=False)
        self.assertEqual(set(frame["id_student"]), {1, 2, 4, 5})
        self.assertAlmostEqual(_row(frame, 2)["clicks_total"], 4)
        self.assertNotIn(3, set(frame["id_student"]))

    def test_future_clicks_and_scores_do_not_change_the_table(self):
        original = build_features(self.ctx, 28, active_only=True)
        mutated = _tables()
        mutated["student_vle"].loc[mutated["student_vle"]["date"] > 28, "sum_click"] = 99999
        mutated["student_assessment"].loc[
            mutated["student_assessment"]["date_submitted"] > 28, "score"
        ] = 0
        other = build_features(prepare(normalize_tables(mutated)), 28, active_only=True)
        columns = feature_columns(original)
        left = original.loc[original["id_student"] == 1, columns].reset_index(drop=True)
        right = other.loc[other["id_student"] == 1, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)

    def test_label_and_unregistration_are_not_features(self):
        frame = build_features(self.ctx, 28, active_only=True)
        columns = feature_columns(frame)
        for banned in (
            "final_result",
            "at_risk",
            "date_unregistration",
            "id_student",
            "code_presentation",
            "imd_group",
            "title",
        ):
            self.assertNotIn(banned, columns)
        self.assertIn("code_module", columns)
        self.assertIn("gender", columns)
        self.assertIn("clicks_total", columns)


class SplitAndMetricTests(unittest.TestCase):
    def test_temporal_split_keeps_years_apart(self):
        frame = pd.DataFrame(
            {
                "code_presentation": ["2013B", "2013J", "2014B", "2014J"],
                "code_module": ["BBB", "AAA", "CCC", "BBB"],
                "id_student": [1, 2, 3, 1],
                "at_risk": [0, 1, 1, 0],
            }
        )
        train, test = temporal_split(frame)
        self.assertEqual(set(train["code_presentation"]), {"2013B", "2013J"})
        self.assertEqual(set(test["code_presentation"]), {"2014B", "2014J"})

    def test_within_module_budget_is_not_the_global_list(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=float)
        modules = np.array(["A"] * 10 + ["B"] * 5)
        y = np.zeros(15, dtype=int)
        y[[9, 8, 7, 14]] = 1
        flag = within_module_flags(modules, scores, budget=0.2)
        self.assertTrue(flag[9] and flag[8] and not flag[7])
        self.assertTrue(flag[14] and not flag[13])
        self.assertAlmostEqual(recall_from_flags(y, flag), 0.75)
        # A global top 20% would spend the whole list inside A.
        global_flag = flags_at_budget(scores, budget=0.2)
        self.assertFalse(global_flag[14])

    def test_group_thresholds_target_recall(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.1, 0.4, 0.2])
        groups = np.array(["A", "A", "A", "A"])
        chosen = thresholds_for_recall(y, scores, groups, target_recall=0.5, min_positives=1)
        flag = apply_group_thresholds(scores, groups, chosen)
        self.assertTrue(flag[0] and not flag[1])

    def test_fnr_matches_fairlearn(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.05, 0.7, 0.6, 0.2, 0.01])
        groups = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
        flag = flags_at_budget(scores, budget=0.5)
        table = group_table(y, flag, groups)
        gap = fnr_gap(table, levels=None, min_n=1, min_pos=1)
        self.assertAlmostEqual(gap, 1.0)
        check_against_fairlearn(y, flag, groups)

    def test_random_split_is_stratified(self):
        frame = pd.DataFrame(
            {
                "code_module": ["AAA"] * 100,
                "code_presentation": ["2014J"] * 100,
                "id_student": np.arange(100),
                "at_risk": [0] * 50 + [1] * 50,
            }
        )
        train, test = random_split(frame, seed=0, test_size=0.3)
        self.assertEqual(len(test), 30)
        self.assertAlmostEqual(test["at_risk"].mean(), 0.5)


class ModelApiTests(unittest.TestCase):
    def _frame(self, n=80):
        return pd.DataFrame(
            {
                "code_module": ["AAA", "BBB"] * (n // 2),
                "code_presentation": ["2013J"] * n,
                "id_student": np.arange(n),
                "gender": ["F", "M"] * (n // 2),
                "region": ["London", "Scotland", "Wales", "Ireland"] * (n // 4),
                "highest_education": ["A Level or Equivalent"] * n,
                "imd_band": ["0-10%", "90-100%"] * (n // 2),
                "age_band": ["0-35", "35-55"] * (n // 2),
                "disability": ["N", "Y"] * (n // 2),
                "num_of_prev_attempts": np.arange(n) % 3,
                "studied_credits": 60,
                "date_registration": -10 - (np.arange(n) % 5),
                "clicks_total": np.arange(n),
                "clicks_last_7d": np.arange(n) % 7,
                "asmt_mean_score": [np.nan if i % 4 == 0 else 40 + (i % 50) for i in range(n)],
                "at_risk": [0] * (n // 2) + [1] * (n // 2),
                "final_result": ["Pass"] * (n // 2) + ["Fail"] * (n // 2),
            }
        )

    def test_both_models_return_probabilities(self):
        frame = self._frame()
        columns = feature_columns(frame)
        train, test = frame.iloc[:60], frame.iloc[60:]
        params = {
            "n_estimators": 12,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "verbosity": -1,
            "n_jobs": 1,
        }
        logistic = fit_logistic(train, test, columns)
        boosted, details = fit_lgbm(train, test, columns, params, return_details=True)
        self.assertEqual(len(logistic), len(test))
        self.assertEqual(len(boosted), len(test))
        self.assertTrue(np.isfinite(logistic).all())
        self.assertTrue(np.isfinite(boosted).all())
        self.assertTrue(((logistic >= 0) & (logistic <= 1)).all())
        self.assertTrue(((boosted >= 0) & (boosted <= 1)).all())
        import shap

        explainer = shap.TreeExplainer(details["model"])
        values = explainer.shap_values(details["X_test"].head(8))
        if isinstance(values, list):
            values = values[-1]
        self.assertEqual(len(np.asarray(getattr(values, "values", values))), 8)

    def test_hyperparameter_grid_runs(self):
        frame = self._frame(120)
        frame.loc[:59, "code_presentation"] = "2013B"
        frame.loc[60:, "code_presentation"] = "2013J"
        frame.loc[:29, "at_risk"] = 0
        frame.loc[30:59, "at_risk"] = 1
        frame.loc[60:89, "at_risk"] = 0
        frame.loc[90:, "at_risk"] = 1
        best = select_hyperparams(frame.iloc[:60], frame.iloc[60:], feature_columns(frame))
        self.assertIn("n_estimators", best)
        self.assertGreaterEqual(best["n_estimators"], 50)
        self.assertGreater(best["val_pr_auc"], 0)


if __name__ == "__main__":
    unittest.main()
