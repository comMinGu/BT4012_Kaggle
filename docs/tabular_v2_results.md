# Tabular v2 — rolling temporal validation

## Outcome

Selected **E04: XGBoost without `time_step`**. It uses exactly `feat_1`–`feat_165`
and labelled observations. Latest-fold AUC is **0.997495**,
mean rolling AUC **0.977915**, and population standard deviation
**0.026357**. No graph features, pseudo-labels, random primary folds,
outer-fold early stopping, or hyperparameter search were used.

The new submission is `outputs/submission_tabular_v2.csv`. It has 15,329 rows,
exactly `index,target`, template-matching keys/order, no duplicate indices or
missing/nonfinite scores, and probabilities in [0,1]. It has not been submitted
automatically. A public Kaggle score for this configuration is **unknown**.

## Preserve E01 and compare like-for-like

E01 remains the original Logistic Regression with 165 features: local AUC
**0.917141**, user-reported public Kaggle AUC **0.88776**. Its notebook, original
metrics JSON, and submission were fingerprinted and preserved. The public/local
gap is approximately **0.029381**. The user's approximate competition benchmark
of 0.92 has not been independently verified.

E01-R evaluates the identical Logistic Regression specification on all three new
folds, without changing E01. Its latest fold reproduces E01 to numerical precision.
It supplies the appropriate rolling reference for comparison; E01 itself retains
blank fold-1/fold-2/rolling-mean fields because those were not part of its original
experiment. New boosted Kaggle values remain blank in the experiment log.

## Fixed rolling folds

| fold | fit_end | fit_rows | fit_0 | fit_1 | validation_start | validation_end | validation_rows | validation_0 | validation_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 18889 | 17118 | 1771 | 21 | 25 | 5311 | 4745 | 566 |
| 2 | 25 | 24200 | 21863 | 2337 | 26 | 30 | 2705 | 2088 | 617 |
| 3 | 30 | 26905 | 23951 | 2954 | 31 | 35 | 4330 | 3640 | 690 |

All rows are labelled; 110,537 unlabelled observations are excluded. Fitting
windows expand while validation windows are disjoint and strictly later. The
validation positive shares are 10.66%, 22.81%, and 15.94%. Changing prevalence alone
does not mechanically explain changes in ROC-AUC; it signals that period
composition differs and affects threshold metrics.

## All measured scores

| ID | Model | Time feature | Fold 1 | Fold 2 | Latest / fold 3 | Mean | Std (ddof=0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E01-R | Logistic Regression | No | 0.894508 | 0.933691 | 0.917141 | 0.915113 | 0.016060 |
| E02 | LightGBM | No | 0.921677 | 0.994402 | 0.998148 | 0.971409 | 0.035199 |
| E03 | LightGBM | Yes | 0.931117 | 0.994504 | 0.998137 | 0.974586 | 0.030773 |
| E04 | XGBoost | No | 0.940656 | 0.995593 | 0.997495 | 0.977915 | 0.026357 |
| E05 | XGBoost | Yes | 0.939144 | 0.995304 | 0.997513 | 0.977320 | 0.027010 |

Mean is the arithmetic mean of the three AUCs, with equal fold weights. Standard
deviation uses `ddof=0` and describes observed fold spread; it is not an error bar
or confidence interval. Do not compare a rolling mean directly with E01's single
latest-period score. Latest-fold gain over E01 is **+0.080354**;
rolling mean gain over E01-R is **+0.062801**.

## Including time_step

Deltas below are `with time - without time`:

| model | delta_fold_1_auc | delta_fold_2_auc | delta_fold_3_auc | delta_mean_auc | delta_std_auc |
| --- | --- | --- | --- | --- | --- |
| LightGBM | +0.009440 | +0.000102 | -0.000011 | +0.003177 | -0.004426 |
| XGBoost | -0.001512 | -0.000289 | +0.000018 | -0.000594 | +0.000653 |

- **LightGBM:** adding time increases mean AUC by 0.003177 and lowers fold spread,
  primarily through a 0.009440 gain on fold 1. Its latest-fold change is -0.000011,
  effectively unchanged at this precision. Time is beneficial on average in these
  folds, not an established improvement for future leaderboard performance.
- **XGBoost:** adding time lowers mean AUC by 0.000594 and increases fold spread.
  Its latest-fold improvement is only 0.000018. The feature is slightly harmful
  overall in this single-seed comparison, with negligible latest-period benefit.

A tree's time thresholds cannot extrapolate a new numerical time trend beyond the
fitting range: all later times fall on the future side of every learned threshold.
Time can still alter the interactions learned within past periods. Results reflect
one seed and fixed settings; no significance claim is made for the small differences.

## Prespecified selection

Before evaluation, the notebook fixed this rule:

1. Keep boosted candidates within **0.002 AUC** of the best latest fold.
2. From those, keep candidates within **0.001 AUC** of the best mean.
3. Prefer the lowest fold standard deviation, then latest AUC/mean for ties.

All four boosted candidates pass the latest-period band. E04 and E05 pass the mean
band. E04 then wins with lower standard deviation, higher mean, and stronger fold-1
performance. It sacrifices only **0.000653 AUC** against the absolute
best latest-fold score, avoiding selection solely on that peak. These practical
near-tie bands are not statistical confidence bounds and were not tuned after results.

## Fixed model settings and reproducibility

Both boosted families use 400 rounds, learning rate 0.05, maximum depth 5,
80% row sampling, all feature columns per tree, L2 regularization 1.0, seed 42,
and one numerical thread. LightGBM additionally uses 31 leaves, minimum 40 rows
per leaf, deterministic CPU execution, and column-wise histogram construction.
XGBoost uses CPU `hist` and `min_child_weight=5`.

Both set `scale_pos_weight` to fitting negative / fitting positive counts:
9.665726, 9.355156, and 8.107989 for the three folds. The final model recomputes it
from all 31,235 labelled observations as 7.571625. No validation/test labels enter
that calculation. Infinities are replaced with NaN in working copies; tree-native
missing handling is used without global scaling or imputation. No feature missingness
or infinity was actually present in the raw inputs.

Versions: Python 3.12.4, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1,
LightGBM 4.6.0, XGBoost 3.0.5, threadpoolctl 3.5.0. The only added Python dependencies
are the two model libraries. Full parameters, versions, raw hashes, feature lists,
and per-experiment config IDs are in `experiments/tabular_v2_config.json`.

Weighting and deterministic CPU settings follow the
[LightGBM 4.6 documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)
and [XGBoost 3.0.5 documentation](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html).
Class-weighted scores are suitable for this ranking evaluation; no probability
calibration claim is made.

## Interpretation and limits

The substantial AUC improvement is consistent with nonlinear thresholds and
feature interactions that Logistic Regression's linear score cannot represent.
This is a plausible explanation, not proof that any particular anonymous feature
has a known meaning. No ID, target, or graph column was accidentally included.
The selected model improves on E01-R in every validation fold.

Every model performs materially worse on fold 1 than the latest period. Even the
selected model has a 0.056839 difference between its best and worst folds. The
near-perfect latest-fold AUC therefore does not establish similar performance on
the later Kaggle test or its public subset. Temporal distribution shift, differing
class-conditional feature patterns, sampling, and unknown supplied-feature lineage
remain possible contributors. The same folds were used to select among four
configurations, so there is some remaining selection optimism.

Retain this run and obtain a manual Kaggle result before interpreting leaderboard
improvement. A sensible next tabular analysis is AUC by individual validation time
step (when both classes are present) to locate weak periods. Avoid repeated tuning
against the public leaderboard. Graph feature engineering has not started.
