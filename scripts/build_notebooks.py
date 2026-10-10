"""Write the three analysis notebooks. Run once; the ipynb files are the artifact."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def cell_md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def cell_code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    book = nbf.v4.new_notebook()
    book["cells"] = cells
    book["metadata"] = {
        "kernelspec": {"display_name": "oulad", "language": "python", "name": "oulad"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return book


def write(name: str, book) -> None:
    path = OUT / name
    path.write_text(nbf.writes(book))
    print(path)


BOOT = '''
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

root = Path.cwd().resolve()
if root.name == "notebooks":
    root = root.parent
sys.path.insert(0, str(root / "src"))

from oulad.config import CACHE_VERSION, FIG_DIR, PROC_DIR, REPORT_DIR
'''

eda = notebook([
    cell_md("""
# Early warning on OULAD

A tutor can contact about a fifth of a module. The question is who to put on that list by week 2, 4, 8 or 12, and whether the list misses students in some demographic groups.

The label is binary. **At risk** means the final result is Withdrawn or Fail. Pass and Distinction are the other class. The four-way label stays in the tables for description only.

Data: the original Open University tables (Kuzilek, Hlosta and Zdrahal, 2017, *Scientific Data*), CC BY 4.0. The Kaggle copy adds an AI-written `vle.title` column; this project does not use it. The OU CDN link currently 404s, so the loader takes the same release from the UCI repository and checks the published row counts (32,593 registrations, 10,655,280 click rows, 22 presentations).
"""),
    cell_code(BOOT + """
from oulad.features import LEAKAGE_RULES
from oulad.pipeline import get_context
import json

ctx = get_context()
exclusion = json.loads((REPORT_DIR / "exclusion.json").read_text())
print("registrations", exclusion["n_rows"])
print("distinct students", exclusion["n_students"])
print("never started (unregistered before day 0)", exclusion["n_never_started"])
print("withdrawn label but no unregistration date", exclusion["n_withdrawn_label_but_no_date"])
print("final results", exclusion["final_result_counts"])
for day in (0, 14, 28, 56, 84):
    print(
        f"day {day:>3}: eligible {exclusion[f'n_eligible_day_{day}']:,}  "
        f"already left {exclusion[f'n_already_left_day_{day}']:,}  "
        f"at-risk rate {exclusion[f'base_rate_day_{day}']:.3f}"
    )
"""),
    cell_md("""
## What is allowed into a feature

`date_unregistration` is not a predictor. It only decides who is still on the module. Students who unregistered before day 0 never started, so they are out of every model. Students who have already left by the cutoff are out of the main model too: a tutor does not need a score to know that. A sensitivity fit puts them back, to show how much that inflates accuracy.

Clicks and assessment scores are cumulative only through the cutoff. Exams are dropped. Banked scores from an earlier presentation are treated as known on day 0.
"""),
    cell_code("""
for rule in LEAKAGE_RULES:
    print("-", rule)
print()
print(ctx.cohort[["code_module", "code_presentation", "final_result", "imd_band", "disability", "age_band", "gender"]].head())
"""),
    cell_md("""
## What the cohort looks like

The outcome mix is not the same module to module, which is why a random split that mixes presentations looks better than a split that has to survive the next year. The click curves are restricted to students still registered in that week, so the withdrawn line is not just zeros after they leave. The first tutor-marked assignment is the first milestone a tutor actually sees.
"""),
    cell_code("""
for name in (
    "outcome_mix",
    "weekly_clicks",
    "first_tma",
    "demographics",
    "withdrawal_km",
    "engagement_clusters",
):
    display(Image(filename=str(FIG_DIR / f"{name}.png")))
"""),
    cell_md("""
The cluster figure uses the whole first 12 weeks. It is a description of how study patterns and outcomes go together. Those full-horizon profiles are not features in the early-warning model, because at week 4 the later weeks have not happened.

The withdrawal curve is time-to-unregistration, censored at the end of the presentation for students who stay. It is not a model of failure. Students who fail are still registered; they are the censored ones.
"""),
])

model = notebook([
    cell_md("""
# Time-sliced models

Features at a cutoff are static demographics plus everything the VLE and the assessment calendar have revealed by that day: total clicks, clicks by activity type, clicks in the last seven days, clicks before the start date, days since the last click, assessment scores submitted by then, how many scheduled assessments are already due, and whether the first tutor-marked assignment is missing or late.

Two models, no search beyond a three-point grid. Logistic regression is the linear baseline. LightGBM is the main model. Tree count, learning rate and number of leaves are chosen on 2013J after fitting on 2013B. 2014 is not used for that choice.

Three evaluations:

- **Temporal.** Train on 2013B and 2013J, test on 2014B and 2014J. This is the number to quote.
- **Random row split.** One stratified 70/30 cut of rows. The same presentation, and sometimes the same student, appears on both sides.
- **Leave one module out.** Train on the other six modules, test on the held-out module. This is module transfer, not a time shift. The headline is a weighted average. The shaded band is the best and worst module.

Recall is not "top 20% of the whole university". Each module flags its own top 20%. Otherwise a difficult module would consume the list. PR-AUC and ROC-AUC stay global for a single model, because those scores are on one scale. Leave-one-module-out averages the per-module areas, weighted by test size, for the same reason: scores from seven different models are not comparable as one ranking.
"""),
    cell_code(BOOT + """
from oulad.features import feature_columns
from oulad.metrics import pool_metrics

features = pd.read_parquet(PROC_DIR / f"features_{CACHE_VERSION}_c28_active.parquet")
columns = feature_columns(features)
print(f"{len(features):,} students still enrolled at day 28")
print(f"{len(columns)} features")
print(", ".join(columns))
"""),
    cell_code("""
metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
show = metrics.loc[metrics["split"].isin(["temporal", "random", "lomo", "temporal_seen_modules", "temporal_unseen_modules", "temporal_keep_withdrawn"])]
cols = ["cutoff", "split", "model", "pr_auc", "roc_auc", "recall_at_20", "base_rate", "n_test", "n_train", "student_overlap", "presentation_overlap"]
show = show.loc[:, [c for c in cols if c in show.columns]].sort_values(["model", "split", "cutoff"])
show.round(3)
"""),
    cell_code("""
# Recompute the day-28 temporal LightGBM numbers from the saved scores.
scores = pd.read_parquet(PROC_DIR / "scores" / CACHE_VERSION / "temporal_lgbm_c28.parquet")
stats = pool_metrics(scores)
quoted = metrics.loc[
    (metrics["split"] == "temporal") & (metrics["model"] == "lgbm") & (metrics["cutoff"] == 28)
].iloc[0]
print("recomputed PR-AUC", round(stats["pr_auc"], 4), "table", round(quoted["pr_auc"], 4))
print("recomputed recall@20", round(stats["recall_at_20"], 4), "table", round(quoted["recall_at_20"], 4))
assert abs(stats["pr_auc"] - quoted["pr_auc"]) < 1e-9
assert abs(stats["recall_at_20"] - quoted["recall_at_20"]) < 1e-9
print("student overlap, temporal test rows whose id is also in train:", round(float(scores["student_overlap"].iloc[0]), 3))
print("share of temporal test rows from a presentation seen in train:", round(float(scores["presentation_overlap"].iloc[0]), 3))
"""),
    cell_md("""
The random split shares presentations with the training rows, so it can learn "this run of BBB is hard". The temporal split shares none of the presentation codes. Some student ids still repeat, because people take more than one module; that overlap is reported above and is not removed.

The sensitivity row `temporal_keep_withdrawn` is the day-28 model with students who have already unregistered left in the data. They are easy to spot, because they have stopped clicking and the label is already Withdrawn. That row is not the system. It is the cost of getting the cohort wrong.

`temporal_unseen_modules` is the 2014 test restricted to modules with no 2013 presentation (CCC). `temporal_seen_modules` is the rest.
"""),
    cell_code("""
for name in ("metric_vs_cutoff", "model_comparison", "lomo_by_module", "shap_day28", "importance_day0_vs_day28"):
    display(Image(filename=str(FIG_DIR / f"{name}.png")))
"""),
    cell_md("""
Day 0 has registration, demographics, and clicks from before the start date. It does not yet have an assessment. The gain chart splits features into protected attributes (gender, age, disability, deprivation), proxies (region, prior education), other static fields, engagement, and assessment. If day 0 is mostly protected attributes and proxies, the early score is a description of who enrolled. That is the point the fairness section starts from.
"""),
])

fair = notebook([
    cell_md("""
# Who the list misses

The decision is a fixed budget: the top 20% inside each module. Accuracy is the wrong summary. A missed at-risk student is a false negative, and the gap that matters is the difference in that rate across groups.

Groups: deprivation collapsed to low (0–30%), mid (30–70%) and high (70–100%), plus a Missing band that is mostly students outside England; declared disability; age band; gender. Missing deprivation is shown but kept out of the low/mid/high gap. A group needs at least 50 people and 20 at-risk students before it enters the gap. Smaller groups stay in the table and are drawn in grey.

The same within-module flags are checked against `fairlearn.MetricFrame` on the day-28 deprivation split. The two implementations have to agree.
"""),
    cell_code(BOOT + """
from oulad.fairness import reports_for_frame

groups = pd.read_csv(REPORT_DIR / "fairness_groups.csv")
day28 = groups.loc[groups["cutoff"] == 28, ["attribute", "group", "n", "n_pos", "base_rate", "selection_rate", "fnr", "fpr", "in_gap", "fnr_gap"]]
day28.round(3)
"""),
    cell_code("""
scores = pd.read_parquet(PROC_DIR / "scores" / CACHE_VERSION / "temporal_lgbm_c28.parquet")
recomputed = reports_for_frame(scores)
merged = day28.merge(
    recomputed[["attribute", "group", "fnr"]],
    on=["attribute", "group"],
    suffixes=("_saved", "_live"),
)
assert (merged["fnr_saved"] - merged["fnr_live"]).abs().max() < 1e-9
print("day-28 false-negative rates match the saved table")
gaps = groups.groupby(["cutoff", "attribute"])["fnr_gap"].first().unstack("attribute")
gaps.round(3)
"""),
    cell_code("""
for name in (
    "fairness_fnr_day28",
    "fairness_fnr_day56",
    "fnr_gap_vs_cutoff",
    "calibration_imd_day28",
    "ablation_fnr_gap",
    "mitigation_tradeoff",
):
    display(Image(filename=str(FIG_DIR / f"{name}.png")))
"""),
    cell_md("""
## Taking the attributes out

The ablation refits the temporal model at day 28 and day 56 three times: with the protected fields, without them, and without them plus region and prior education. Engagement and assessment history stay in. If the false-negative gap barely moves, the model is not "using gender" in the simple sense. It is using study behaviour that differs across groups, and a shared threshold then misses different fractions of the at-risk students.

Day 0 is the uncomfortable end of the same plot. Before anyone has clicked, the score is mostly who the student is. A high day-0 PR-AUC is not a reason to send that list to a tutor.
"""),
    cell_code("""
ablation = pd.read_csv(REPORT_DIR / "ablation.csv")
ablation.pivot_table(index=["cutoff", "attribute"], columns="variant", values="fnr_gap").round(3)
"""),
    cell_md("""
## Equalising recall

The mitigation does not retrain the booster. A model is fit on 75% of the 2013 rows. On the other 25% it picks, for each group, the score cutoff that flags about the same fraction of at-risk students. Those cutoffs are applied to 2014. Sweeping the target recall traces the gap against overall recall. The stars are the unmitigated rule on that same model: each module's top 20%.

This is an illustration of the tradeoff, not a policy. Thresholds chosen on a few hundred positives in a small group are noisy, and a group with too few validation positives falls back to the overall cutoff. Nothing here was tuned on 2014.
"""),
    cell_code("""
import json
curves = pd.read_csv(REPORT_DIR / "mitigation.csv")
chosen = curves.loc[curves["chosen"], ["cutoff", "attribute", "target_recall", "test_recall", "selection_rate", "fnr_gap", "n_fallback_groups"]]
baselines = pd.DataFrame(json.loads((REPORT_DIR / "results.json").read_text())["mitigation_baselines"])
print("Unmitigated within-module top 20% on the threshold model")
display(baselines.round(3))
print("Operating point closest to a 20% overall selection rate")
chosen.round(3)
"""),
    cell_md("""
## Limits

Exam scores are missing for most students, so the label is the final result, not a mark. Withdrawn and Fail are different events: one stops early, the other stays and does not pass. A single at-risk label is what a tutor acts on, and it hides that difference.

The click vocabulary is this university's VLE. A model that transfers across these seven modules can still fail on a different institution. The threshold adjustment is not a deployed intervention, and it was not tested as one. Deprivation is a property of the postcode area, not of the person, and it is missing for students outside the English index.
"""),
])

write("01_eda.ipynb", eda)
write("02_model.ipynb", model)
write("03_fairness.ipynb", fair)
