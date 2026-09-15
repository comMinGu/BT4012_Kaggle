# Baseline results

Executed `notebooks/01_baseline.ipynb` from a fresh Python 3.12.4 kernel on
2026-09-15. All 11 code cells completed without errors. Both logistic regressions
converged; no convergence warnings were suppressed.

## Fixed configuration

- Features: `feat_1`–`feat_165` only; no graph information or ID/time predictors.
- Training eligibility: known labels only; 110,537 unlabelled rows excluded.
- Class meaning: user-confirmed 0 = negative, 1 = positive / illicit.
- Pipeline: replace infinities with NaN in memory, median imputation,
  StandardScaler, L2 logistic regression, C=1, lbfgs, balanced class weights,
  max_iter=3000, tol=1e-4, random seed 42, one numerical thread.
- Fit: steps 1–30, 26,905 rows (23,951 negative / 2,954 positive).
- Validation: steps 31–35, 4,330 rows (3,640 negative / 690 positive).
- Final refit: all 31,235 labelled rows in steps 1–35.
- Solver iterations: 191 for validation fit,
  209 for final refit.

## Validation

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.917141265 |
| Precision at 0.5 | 0.342943854 |
| Recall at 0.5 | 0.982608696 |
| F1 at 0.5 | 0.508436445 |

ROC-AUC uses continuous class-1 probabilities. Other metrics use an untuned 0.5
threshold. Confusion matrix: TN=2,341, FP=1,299, FN=12, TP=678. The constant-score
reference has ROC-AUC 0.5. No test ROC-AUC is available.

## Submission and integrity

`outputs/submission_baseline.csv` contains 15,329 rows and exactly `index,target`.
The target is `predict_proba` for class 1. Sample-template indices match in order;
there are no duplicate indices, missing scores, or nonfinite scores. All scores
are in [0,1]. Checks passed before writing and after reading the saved CSV.
Some numerical probabilities saturate at 0 or 1; they are unthresholded model
outputs, not hard predictions. No clipping or arbitrary score changes were applied.

All four raw hashes still match `docs/raw_manifest.json`. No supplied feature
values were missing or infinite. All expected dataset counts/schema observations
were verified. The raw data and all generated output files are excluded from Git.
Detailed aggregate run metadata is regenerated in `outputs/baseline_metrics.json`.

## Interpretation and next experiment

The weighted model finds 678 of 690 validation illicit transactions at threshold
0.5, with 1,299 false positives. This threshold result is distinct from ROC-AUC.
Validation illicit prevalence (15.94%) exceeds fitting prevalence (10.98%), which
reinforces the importance of temporal evaluation. Class weighting can affect
probability calibration; no claim is made that these scores are calibrated risks.

Next compare `class_weight=None` with the same pipeline and time split. Keep this
baseline fixed for comparison. Do not tune repeatedly against one holdout or use
validation/test labels in preprocessing. The supplied anonymous features' creation
history remains unknown; no graph features are used.
