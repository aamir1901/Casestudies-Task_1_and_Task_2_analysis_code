# COSC2669 Case Studies in Data Science

Amir Munawar Wangde (s4186051), RMIT University

Analysis code for Individual Task 1 and Individual Task 2. Both tasks predict
at-risk students from learning analytics data, framed around a Data Scientist
role at the Australian Council for Educational Research (ACER).

## Task 1 — Part 1.3 analysis

`Task1_Part1_3_Analysis.ipynb`

Binary at-risk classification on two datasets, using a random forest and a
multilayer perceptron. Covers preprocessing, feature engineering from the VLE
interaction log, model evaluation on a held-out test set, and feature
importances (`fig_importance.pdf`).

## Task 2 — Part 2 deliberation

`Task2_Part2_Deliberation.ipynb`

Re-examines the Task 1 analysis for evaluation soundness, learning behaviour
and fairness. The models and preprocessing are unchanged, so every difference
comes from the evaluation design. The notebook runs, in order:

1. **Leakage audit** — students with repeated enrolments, students appearing in
   both subject files, and engagement features counted after the outcome.
2. **Cross-validation comparison** — single 80/20 split vs stratified 5-fold vs
   stratified *group* 5-fold, plus a model restricted to week-four features.
3. **Learning curves** — recall, F1 and ROC-AUC against training-set size,
   computed from the same fits across five student-disjoint folds.
4. **Fairness audit** — Fairlearn `MetricFrame` on out-of-fold predictions,
   with recall, FPR and selection rate by sensitive group.
5. **Sensitive-attribute removal** — retraining without them, to test whether
   bias flows through proxies.
6. **Mitigation** — `ThresholdOptimizer` under an equalised-odds constraint.

### Outputs

| File | Contents |
|---|---|
| `cv_comparison.csv` | Metrics under each evaluation design |
| `learning_curves.csv` | Per-fold, per-size scores |
| `learning_curve_summary.csv` | Validation scores at smallest and full size |
| `fairness_summary.csv` | Disparity metrics by sensitive attribute |
| `unawareness_test.csv` | Equalised-odds gaps with and without sensitive inputs |
| `mitigation.csv` | Before and after `ThresholdOptimizer` |
| `fig_learning_curves.pdf` | Learning curves |
| `fig_fairness_oulad.pdf` | Group metrics, OULAD |
| `fig_fairness_studperf.pdf` | Group metrics, Student Performance |

## Data (not included)

Both datasets are public and are not committed here, since the OULAD
interaction log alone exceeds 10 million rows.

- **OULAD** — https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
  (`studentInfo.csv`, `studentVle.csv`)
- **Student Performance** — https://archive.ics.uci.edu/dataset/320/student+performance
  (`student-mat.csv`, `student-por.csv`)

Place the CSVs next to the notebook, or set `DATA` in the setup cell to
wherever they live.

## Requirements

Python 3.11+, with `pandas`, `numpy`, `scikit-learn`, `matplotlib` and
`fairlearn`. The Task 2 notebook installs `fairlearn` on first run if it is
missing. A full run takes roughly 15–30 minutes; set `FAST = True` in the setup
cell for a quicker draft run with fewer trees and fewer training sizes.
