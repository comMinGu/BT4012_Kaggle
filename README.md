# BT4012_Kaggle

Bitcoin Fraud Analytics — individual BT4012 university Kaggle competition.
**Metric: ROC-AUC. Class 0 = negative; class 1 = positive / illicit.**

## Current results

E01 Logistic Regression is preserved: local AUC **0.917141**, public Kaggle AUC
**0.88776** (reported by the user). The latest experiment evaluates LightGBM and
XGBoost with and without time across three rolling temporal folds.

| ID | Model | Time feature | Fold 1 | Fold 2 | Latest / fold 3 | Mean | Std (ddof=0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E01-R | Logistic Regression | No | 0.894508 | 0.933691 | 0.917141 | 0.915113 | 0.016060 |
| E02 | LightGBM | No | 0.921677 | 0.994402 | 0.998148 | 0.971409 | 0.035199 |
| E03 | LightGBM | Yes | 0.931117 | 0.994504 | 0.998137 | 0.974586 | 0.030773 |
| E04 | XGBoost | No | 0.940656 | 0.995593 | 0.997495 | 0.977915 | 0.026357 |
| E05 | XGBoost | Yes | 0.939144 | 0.995304 | 0.997513 | 0.977320 | 0.027010 |

**Selected: E04, XGBoost without `time_step`.** It combines a latest AUC within
0.002 of the best with the strongest mean and lowest boosted-model fold spread.
Its new Kaggle score is unknown; local AUC is not a leaderboard prediction.
See [full results and interpretation](docs/tabular_v2_results.md) and the
[experiment log](experiments/results.csv).

## Layout

```text
data/raw/                                  Immutable competition CSVs; ignored
notebooks/01_baseline.ipynb                  Preserved E01 Logistic Regression
notebooks/02_boosted_tree_experiments.ipynb   Rolling tabular experiments and submission
experiments/                               Versionable aggregate results/configurations
scripts/audit_data.py                       Detailed read-only dataset audit
src/                                       Helpers only when reuse warrants them
outputs/                                   Predictions and generated run metadata; ignored
docs/                                      Reviewed data audit and experiment reports
requirements.txt                           Pinned direct project dependencies
```

The original raw files were relocated from `Data/` to `data/raw/` without changing
their SHA-256 hashes. Place the four downloaded CSVs there after cloning. The
parent directory name does not affect execution.

## Dataset

| File | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| `train.csv` | 141,772 | 168 | `txId`, `time_step`, 165 anonymized features, nullable `label` |
| `test.csv` | 15,329 | 168 | `index`, `txId`, `time_step`, same 165 features |
| `txs_edgelist.csv` | 234,355 | 2 | Directed endpoint convention `txId1 -> txId2` |
| `sample_submission.csv` | 15,329 | 2 | `index,target` placeholder template |

There are **31,235 labelled training rows**: 27,591 negative and 3,644 positive
(11.6664% positive), plus 110,537 unlabelled rows excluded from all model fitting.
Train covers steps 1–35; test covers 36–49. No missing/infinite features or duplicate
transaction IDs were found. Thirteen identical-feature groups cover 44 distinct
training transactions; these were retained. The graph has 46,668 endpoints without
supplied feature rows. No graph information is used in the models.

Feature meanings, construction history, and the calendar duration of a time step
remain undocumented. [Data audit](docs/data_audit.md) and
[raw manifest](docs/raw_manifest.json) provide details.

## Evaluation and models

| Fold | Fitting period | Validation period | Fitting rows | Validation rows |
| --- | --- | --- | ---: | ---: |
| 1 | 1–20 | 21–25 | 18,889 | 5,311 |
| 2 | 1–25 | 26–30 | 24,200 | 2,705 |
| 3 | 1–30 | 31–35 | 26,905 | 4,330 |

Use labelled rows only and identical masks across models. Mean is unweighted;
standard deviation uses `ddof=0`. It describes three-fold variation, not uncertainty
in future performance. The original E01 holdout corresponds to fold 3; E01-R
reevaluates its unchanged specification across all three folds in notebook 02.

Boosted models use 400 rounds, learning rate 0.05, shallow trees, seed 42, one
numerical thread, and fitting-only negative/positive class-ratio weights. There is
no outer-fold early stopping or parameter search. Infinities become NaN in working
copies; trees use native missing handling. Features are `feat_1`–`feat_165`, plus
`time_step` only in E03/E05. IDs and `label` are always excluded. Unlabelled rows,
validation labels, and test data never fit preprocessing or class weights.

The prespecified selection rule prioritizes latest AUC within a 0.002 near-tie
band, then mean within 0.001, then lower fold spread. E04 is refitted on all labelled
steps 1–35 without retuning. No GNN, graph features, pseudo-labels, or random primary
cross-validation are used. See [experiment conventions](experiments/README.md).

## Install and execute

Use **Python 3.12** (verified on 3.12.4):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name bt4012-baseline --display-name "BT4012 baseline"
jupyter lab notebooks/02_boosted_tree_experiments.ipynb
```

Choose **BT4012 baseline**, restart the kernel, and run all cells. The kernel is
installed inside the environment. If `python` already points to Python 3.12, use
it to create the environment. For a fresh headless execution:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=bt4012-baseline \
  --ExecutePreprocessor.timeout=900 notebooks/02_boosted_tree_experiments.ipynb
```

Notebook 02 does not require rerunning or overwriting notebook 01. It preserves
any existing E01 output artifacts. The audit can still run separately:

```bash
python scripts/audit_data.py
```

Experiments use in-memory data and CPU fitting. `outputs/tabular_fold_progress.csv`
records aggregate progress as each fold completes. A successful notebook execution
ends with the final submission checks and experiment-log update; partial progress
alone is not a completed run.

## Outputs and recording

- `outputs/submission_baseline.csv`: original E01 submission, if already generated.
- **`outputs/submission_tabular_v2.csv`**: selected model's class-1 probabilities,
  15,329 rows, exactly `index,target`, in sample-template order.
- `outputs/tabular_v2_metrics.json`: complete aggregate run/selection metadata.
- `experiments/results.csv`: persistent E01/E01-R/E02–E05 comparison and manual
  Kaggle-score fields; new Kaggle scores remain blank until reported.
- Other `experiments/` CSV/JSON files: fold counts, fold scores, time comparison,
  and exact reproducibility configuration.

Submission checks cover columns, row count, exact keys/order, index uniqueness,
missingness, finite probabilities in [0,1], and CSV reload. Re-running the same
experiment updates only its own aggregate rows and tabular-v2 output. Changed
configurations require new experiment IDs. No code submits to Kaggle automatically.

## Repository safety

Existing remote: **https://github.com/comMinGu/BT4012_Kaggle.git**. It is private;
keep it private unless explicitly instructed otherwise. Preserve existing remote
history and never force-push. Raw CSVs, prediction outputs, environments, caches,
notebook checkpoints, OS metadata, secrets, and local agent notes are ignored.
Review staged contents before committing; `.gitignore` cannot protect secrets
pasted into source or previously tracked files. Notebook outputs contain aggregates
only. No raw records or individual predictions should be committed.

## Interpretation and next step

Boosted trees improve all three folds relative to the linear reference, consistent
with nonlinear thresholds and feature interactions. Time helps LightGBM's mean
but slightly harms XGBoost's mean/stability; its latest-period effect is negligible
for both. Every model still performs worst in fold 1, so a high latest AUC does not
resolve temporal generalization concerns. See the [results report](docs/tabular_v2_results.md).

Keep the configurations fixed, record the next manually obtained Kaggle score, and
inspect per-time-step validation AUC to locate weak periods. No graph engineering yet.
