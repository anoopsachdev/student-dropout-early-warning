"""Paths, cohort rules, and column roles shared across the analysis."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# Bump when feature or model definitions change so cached parquet is not reused.
CACHE_VERSION = "v1"

CUTOFFS = (0, 14, 28, 56, 84)
BUDGET = 0.20
RANDOM_SEED = 0

KEYS = ["code_module", "code_presentation", "id_student"]
TRAIN_PRESENTATIONS = ("2013B", "2013J")
TEST_PRESENTATIONS = ("2014B", "2014J")
AT_RISK_RESULTS = ("Withdrawn", "Fail")

IMD_LOW = ("0-10%", "10-20%", "20-30%")
IMD_MID = ("30-40%", "40-50%", "50-60%", "60-70%")
IMD_HIGH = ("70-80%", "80-90%", "90-100%")
IMD_ORDER = [f"{i}-{i + 10}%" for i in range(0, 100, 10)] + ["Missing"]
IMD_GROUP_ORDER = ("low", "mid", "high", "Missing")

PROTECTED_FEATURES = ("gender", "age_band", "disability", "imd_band")
PROXY_FEATURES = ("region", "highest_education")
GROUP_COLUMNS = ("imd_group", "disability", "age_band", "gender")

# code_module is a feature. Presentation and student id are not.
NON_FEATURES = {
    "code_presentation",
    "id_student",
    "final_result",
    "at_risk",
    "date_unregistration",
    "imd_group",
    "cutoff",
    "module_length",
    "title",
}

CATEGORICAL_FEATURES = {
    "code_module",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
}

STATIC_FEATURES = {
    "code_module",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
}

ASSESSMENT_FEATURES = {
    "asmt_mean_score",
    "asmt_min_score",
    "n_due",
    "n_submitted_due",
    "submit_rate",
    "first_tma_due",
    "first_tma_missing",
    "first_tma_days_late",
    "first_tma_score",
}

OUTCOME_ORDER = ("Distinction", "Pass", "Fail", "Withdrawn")
OUTCOME_COLORS = {
    "Distinction": "#009E73",
    "Pass": "#56B4E9",
    "Fail": "#E69F00",
    "Withdrawn": "#D55E00",
}
LINE_COLORS = {
    "temporal": "#0072B2",
    "random": "#E69F00",
    "lomo": "#009E73",
    "logistic": "#CC79A7",
    "lgbm": "#0072B2",
}

# Groups smaller than this stay in the tables but do not enter the FNR gap.
MIN_GROUP_N = 50
MIN_GROUP_POS = 20

EXPECTED_ROWS = {
    "student_info": 32593,
    "student_vle": 10655280,
    "courses": 22,
}
