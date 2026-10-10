# Early warning on OULAD, with a fairness audit

By a chosen day of a module, a tutor can contact about a fifth of the students still enrolled. This project ranks who should be on that list, and checks whether the list misses at-risk students in some groups more than others.

The label is binary. **At risk** means the final result is Withdrawn or Fail. Pass and Distinction are the other class. The four-way result is kept for the descriptive plots only.

The number to quote is the **temporal** split: train on the 2013 presentations, test on 2014. At day 28, LightGBM reaches a PR-AUC of 0.722 and finds 35.5% of the at-risk students when each module flags its top 20%. A random split of the same rows looks better (PR-AUC 0.778) because it has seen those presentations before. The deprivation gap in missed students is 0.13 at day 28, down from 0.28 on day 0, and it is mostly the less-deprived at-risk students who are missed.

## Data

The tables are the original Open University Learning Analytics Dataset (Kuzilek, Hlosta, and Zdrahal, *Scientific Data* 4:170171, 2017, CC BY 4.0): 32,593 registrations, 28,785 students, 10,655,280 VLE click rows, and 22 module presentations. The Open University download links now 404, so `scripts/download_oulad.py` uses the UCI mirror of the same release. Row counts are checked on load (32,593 student-info rows, 10,655,280 VLE rows, 22 courses).

Do not use a Kaggle extract that adds `vle.title`. That column is placeholder text and is dropped if it appears.

Final results in the file: Pass 12,361, Withdrawn 10,156, Fail 7,052, Distinction 3,024.

### Who is in the model

`date_unregistration` is an eligibility rule, not a feature.

- 2,678 students unregistered before day 0. They never started and are out of every cutoff.
- The main model also drops anyone who has already left by the cutoff. A tutor does not need a model to know that. At day 28 that leaves 27,538 students, of whom 44.1% finish at risk.
- A sensitivity fit at day 28 puts those already-withdrawn students back in. PR-AUC rises from 0.722 to 0.811. That inflation is the cost of scoring people who have already left. Outreach recall barely moves (0.355 to 0.359), because the extra cases are easy and the budget is still 20% inside each module.
- 93 rows are labelled Withdrawn but have no unregistration date. They stay in, because the date rule cannot see them.
- 1,909 banked assessment scores are treated as known from day 0, so a transfer date cannot hide them.

Eligible cohort and base rate, students still enrolled:

| Cutoff | Students | At-risk rate |
| --- | ---: | ---: |
| Day 0 | 29,496 | 0.478 |
| Day 14 | 28,119 | 0.453 |
| Day 28 | 27,538 | 0.441 |
| Day 56 | 26,522 | 0.420 |
| Day 84 | 25,724 | 0.402 |

The base rate falls because the people who leave early are disproportionately at risk. Later models are asked a harder question about a smaller cohort.

### What a feature is allowed to know

Everything time-dependent is built inside `build_features(cutoff)`.

- Clicks after the cutoff are excluded. Totals are the cumulative sum through the last day at or before the cutoff.
- The last-seven-day window is the seven days ending on the cutoff.
- Clicks before day 0 are a separate pre-start total.
- Assessment scores count only when `date_submitted` is at or before the cutoff. Exams are excluded: they are the end of the module and are missing for most students.
- An assessment is "due" only when its scheduled date is at or before the cutoff. That calendar is known in advance.
- `final_result` is the label only. `id_student` and `code_presentation` are not features. `code_module` is a feature. `imd_group` is used in the audit and is not a model input.

Static fields are demographics, previous attempts, studied credits, and registration date. Engagement fields are click totals, active days, clicks by activity type, days since the last click, recent clicks, and pre-start clicks. Assessment fields are mean and minimum score so far, submissions against assessments due, and lateness or absence of the first tutor-marked assignment.

## Evaluation

Three ways of holding students out, at cutoffs 0, 14, 28, 56, and 84:

- **Temporal.** Train on 2013B and 2013J, test on 2014B and 2014J. No test presentation appears in training. About 9% of day-28 test rows are a student id that also appears in training, on a different module. That overlap is reported and not removed.
- **Leave-one-module-out.** Each module is scored by a model trained on the other modules, mixing 2013 and 2014. Scores are not pooled across folds, because the scales differ. PR-AUC and ROC-AUC are averaged with weights proportional to test size; recall is weighted by the number of at-risk students.
- **Random.** One stratified 70/30 split of rows, seed 0, not grouped by student. Every test presentation also appears in training. This split is here to show the inflation.

The outreach rule is the top 20% **inside each module**, not the top 20% of the university. A hard module cannot take the whole list, and a deprivation gap is not just a module-mix gap. Recall is true positives over at-risk students under that rule. PR-AUC and ROC-AUC are global for one model.

The baseline is logistic regression (median imputation, scaling, one-hot encoding, balanced class weights, log1p on click counts). The main model is LightGBM with native missing values and an ordinal encoding that maps unseen categories, including a module never seen in training, to missing. A three-point search on 2013B against 2013J at day 28, with early stopping, selected 63 leaves, learning rate 0.05, and 50 trees. Every cell of that search hit the 50-tree floor, so the same small model is frozen for every cutoff and every split. It is not retuned per fold. The validation PR-AUC stored in `reports/hyperparams.json` (0.546) is the early-stopped fit, not the 50-tree model used for the test numbers below.

## Results

LightGBM on the 2014 presentations:

| Day | Test rows | Base rate | PR-AUC | ROC-AUC | Recall at 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 17,195 | 0.487 | 0.675 | 0.688 | 0.300 |
| 14 | 16,027 | 0.450 | 0.670 | 0.711 | 0.323 |
| 28 | 15,729 | 0.440 | 0.722 | 0.762 | 0.355 |
| 56 | 15,092 | 0.416 | 0.764 | 0.800 | 0.403 |
| 84 | 14,590 | 0.396 | 0.790 | 0.831 | 0.442 |

A random 20% list would recall 0.20. By day 28 the within-module list recalls 0.355 of at-risk students. The day-14 PR-AUC is slightly below day 0 (0.670 against 0.675) while ROC-AUC rises (0.711 against 0.688): the easiest withdrawals have already left, and PR-AUC moves with the base rate.

Logistic regression is close. At day 28 its PR-AUC is 0.700 and its recall is 0.352, against 0.722 and 0.355 for LightGBM. At day 0 and day 84 the linear model is slightly ahead on PR-AUC (0.688 vs 0.675, and 0.797 vs 0.790). The trees are not buying much once the features are this simple.

The random split at day 28 scores PR-AUC 0.778 and recall 0.374. It shares every presentation with training and 14% of its student ids. That is the optimistic number.

Leave-one-module-out at day 28 averages PR-AUC 0.724 and recall 0.346, in line with the temporal result, but the modules do not agree. Held-out PR-AUC runs from 0.50 on GGG and 0.55 on AAA to 0.82 on CCC and DDD. A single cross-module number hides that.

CCC has no 2013 presentation, so it is unseen in the temporal training set. At day 28 its PR-AUC is 0.824 and its recall is 0.344, against 0.687 and 0.359 on the modules the model has seen. CCC's base rate is also higher (0.523 against 0.415), so the higher PR-AUC is not pure evidence of easy transfer. ROC-AUC is higher too (0.813 against 0.747).

SHAP on the day-28 temporal model points at submission rate, assessment scores, prior education, and active days. Split gain tells the same story over time. At day 0, 52% of the gain is engagement (mostly clicks before the start), 24% is other static fields such as module and credits, 15% is region and prior education, and 9% is the protected attributes. There is no assessment yet. At day 28, assessment features are 48% of the gain, engagement 34%, and the protected attributes 3%.

Figures: `reports/figures/metric_vs_cutoff.png`, `model_comparison.png`, `lomo_by_module.png`, `shap_day28.png`, `importance_day0_vs_day28.png`.

## Fairness

Groups are deprivation (IMD bands collapsed to low 0–30%, mid 30–70%, high 70–100%; Missing is reported and left out of the gap), declared disability, age band, and gender. A group enters the gap only with at least 50 students and 20 at-risk students. The harm metric is the false-negative rate: at-risk students the top-20% list does not contact. The gap is the largest rate minus the smallest, inside the levels that qualify.

Day-28 temporal LightGBM, 2014 test:

| Group | Level | Students | At risk | Selection rate | FNR |
| --- | --- | ---: | ---: | ---: | ---: |
| Deprivation | low | 4,768 | 2,450 | 0.275 | 0.582 |
| Deprivation | mid | 6,231 | 2,704 | 0.192 | 0.654 |
| Deprivation | high | 4,119 | 1,579 | 0.140 | 0.713 |
| Disability | N | 14,239 | 6,109 | 0.195 | 0.648 |
| Disability | Y | 1,490 | 805 | 0.242 | 0.625 |
| Age | 0–35 | 10,943 | 5,018 | 0.217 | 0.628 |
| Age | 35–55 | 4,668 | 1,854 | 0.162 | 0.687 |
| Age | 55<= | 118 | 42 | 0.093 | 0.857 |
| Gender | F | 6,668 | 2,751 | 0.186 | 0.685 |
| Gender | M | 9,061 | 4,163 | 0.210 | 0.618 |

The deprivation gap is 0.131. The list contacts low-IMD students more often, which matches their higher base rate (0.514 against 0.383). The students it misses are the at-risk students in the high-IMD band. On day 0 that same gap is 0.284 (FNR 0.543 low, 0.828 high). It shrinks once scores and clicks exist, and it is still 0.105 at day 84.

Disability is nearly balanced (gap 0.023). Gender is a smaller gap (0.067), with women missed more often. The age gap is 0.230 and is carried by the 55<= band, which has 42 at-risk students in this test slice. That rate is unstable.

Low, mid, and high deprivation bands are reasonably calibrated at day 28: predicted risk tracks the observed at-risk rate. The Missing band, often students outside England, sits below the diagonal in the middle of the score range.

### Dropping the attributes

Refitting the temporal model without gender, age, disability, and IMD band, and then also without region and prior education, barely changes ranking. Day-28 PR-AUC goes from 0.722 to 0.720 to 0.719. The gaps do move.

| Gap at day 28 | Full model | Drop protected | Also drop proxies |
| --- | ---: | ---: | ---: |
| Deprivation | 0.131 | 0.063 | 0.037 |
| Age | 0.230 | 0.102 | 0.146 |
| Disability | 0.023 | 0.018 | 0.005 |
| Gender | 0.067 | 0.069 | 0.060 |

Removing the deprivation and age fields shrinks those gaps. Gender barely changes, so that gap is coming from study behaviour under a shared cutoff. At day 56 the deprivation gap only moves from 0.126 to 0.087 after both drops, and the gender gap does not shrink.

### Equalising recall

This is a different fit from the table above. The booster is trained on 75% of the 2013 rows. On the other 25% it picks, per group, the score cutoff that flags a target fraction of at-risk students. Those cutoffs are applied to 2014. Groups with fewer than 20 validation positives keep the global cutoff. The stars on `reports/figures/mitigation_tradeoff.png` are the unmitigated within-module top 20% of this same model, not of the all-2013 model.

At day 28 that unmitigated list has recall 0.356 and these gaps: deprivation 0.119, age 0.160, gender 0.058, disability 0.033.

The operating point is the smallest test gap among selection rates between 15% and 25%:

| Day 28 | Recall | Selection | FNR gap |
| --- | ---: | ---: | ---: |
| Deprivation, group thresholds | 0.394 | 0.232 | 0.040 |
| Age, group thresholds | 0.381 | 0.220 | 0.127 |
| Gender, group thresholds | 0.417 | 0.247 | 0.028 |
| Disability, group thresholds | 0.373 | 0.219 | 0.223 |

Group thresholds cut the deprivation gap from 0.119 to 0.040 and the gender gap from 0.058 to 0.028, with a higher overall recall and a selection rate a bit above 20%. They do not help disability. The shared list was already close, and the cutoffs chosen on 2013 open a 0.22 gap on 2014. That is why this plot is an illustration of the tradeoff, not a threshold to deploy. The age result is also fragile: the 55<= band is small.

## What the descriptive plots add

Outcome mix differs sharply by module (`reports/figures/outcome_mix.png`). Weekly clicks, counted only among students still registered that week, already separate Distinction, Pass, Fail, and Withdrawn (`weekly_clicks.png`). The first tutor-marked assignment is higher for students who pass, and 46% of students who withdraw never submit it (`first_tma.png`).

Two plots are not features. A Kaplan–Meier curve of time to withdrawal, censored at the module length, shows a higher withdrawal hazard in the low-IMD band (`withdrawal_km.png`). A four-cluster summary of the first 12 weeks of clicks, fit on the full horizon, separates a sustained high-click profile from profiles that fade or stop early (`engagement_clusters.png`). Using those clusters in the model would leak the future.

## Limitations

- Most students have no exam score in the file. The label is the recorded final result, not a mark.
- Withdrawn and Fail are different events. One stops early, the other stays and does not pass. A single at-risk label is what a contact list acts on, and it hides that difference. The sensitivity fit is the quantitative version of this warning.
- The VLE activity types are specific to this platform. A model that leans on `oucontent` clicks will not transfer to another university's logs.
- The group-threshold mitigation was not tuned on 2014, and it is not a deployment recommendation. On disability it makes the test gap worse.
- The age gap depends on a band with a few dozen at-risk students in the test year.
- About 9% of temporal test rows reuse a student id from training, on another module.
- The booster is intentionally small (50 trees). A larger search might move the metrics. It would not change the split design or the direction of the gaps.
- CCC is the only module absent from 2013, so "unseen module" is one course with a high base rate, not a general transfer test. Leave-one-module-out is the broader check, and GGG is where it fails.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_oulad.py
mkdir -p .mpl
MPLCONFIGDIR="$PWD/.mpl" PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py
```

Skip the download if `data/raw/oulad_uci.zip` or the seven CSVs are already present. `data/raw/` and `data/processed/` are gitignored. The pipeline writes `reports/metrics.csv`, `reports/fairness_groups.csv`, `reports/ablation.csv`, `reports/mitigation.csv`, `reports/results.json`, and the figures under `reports/figures/`.

`--stage eda`, `--stage models`, or `--stage fairness` reruns one part. `--rebuild` ignores feature and score caches.

Notebooks read those artifacts and recompute the day-28 metrics from the saved scores:

```bash
.venv/bin/python -m ipykernel install --sys-prefix --name oulad --display-name oulad
mkdir -p .ipython
JUPYTER_PATH="$PWD/.venv/share/jupyter" IPYTHONDIR="$PWD/.ipython" \
MPLCONFIGDIR="$PWD/.mpl" PYTHONPATH=src .venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/01_eda.ipynb notebooks/02_model.ipynb notebooks/03_fairness.ipynb \
  --ExecutePreprocessor.kernel_name=oulad --ExecutePreprocessor.timeout=1800
```

Tests, including a synthetic check that future clicks and future scores do not change features:

```bash
MPLCONFIGDIR="$PWD/.mpl" OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_core -v
```
