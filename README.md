# BT4012_Kaggle

Bitcoin Fraud Analytics — BT4012

Individual university Kaggle competition: classify Bitcoin transactions as illicit
or licit using transaction features and a payment-flow graph. **ROC-AUC** is the
competition metric. The user confirms **0 = negative, 1 = positive / illicit**.

The first end-to-end baseline has been executed successfully. It uses only the
165 anonymized transaction features and labelled observations. No graph features,
GNN, pseudo-labels, or hyperparameter search are used. Nothing has been submitted
to Kaggle automatically.

## Repository structure

```text
data/raw/                    Four unchanged competition CSVs; ignored by Git
notebooks/01_baseline.ipynb   Executed baseline, with 11 clear sections
src/                         Reserved for helpers when reuse is needed
scripts/audit_data.py         Standalone detailed data audit
outputs/                     Generated submission and aggregate run record; ignored
docs/data_audit.md            Reviewed detailed audit summary
docs/raw_manifest.json        Original file hashes and current raw location
docs/baseline_results.md      Baseline configuration, results, and limitations
requirements.txt             Pinned direct project dependencies
```

The raw files were relocated from `Data/` to `data/raw/` without changing their
SHA-256 hashes. Download/place the four competition files there when cloning the
repository. Raw files, predictions, credentials, and local environments are not
part of the repository. The parent folder need not have a specific name.

## Dataset files and verified structure

| File | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| `train.csv` | 141,772 | 168 | `txId`, `time_step`, `feat_1`–`feat_165`, nullable `label` |
| `test.csv` | 15,329 | 168 | `index`, `txId`, `time_step`, same 165 features |
| `txs_edgelist.csv` | 234,355 | 2 | Directed endpoint convention `txId1 -> txId2` |
| `sample_submission.csv` | 15,329 | 2 | `index,target`; placeholder target 0.5 |

Training has **31,235 labelled rows**: 27,591 class 0 and 3,644 class 1
(**11.6664% illicit**). Another 110,537 rows have unknown labels and are excluded
from model fitting. Train time steps are 1–35; test time steps are 36–49.
All values/counts above were checked against the actual CSVs, not assumed.

No feature missingness or infinity, duplicate transaction IDs, or train/test ID
overlap was found. Submission keys match. There are 13 groups covering 44 distinct
training transactions with identical features and no conflicting known labels;
all are retained. The full graph contains 203,769 nodes, of which 46,668 have no
supplied feature row. The edge list is inspected but never used by this baseline.
Anonymous feature meanings, construction history, and time-step duration remain
undocumented. See the [detailed audit](docs/data_audit.md).

## Baseline and validation

Features are exactly `feat_1`–`feat_165`; exclude `txId`, `index`, `label`, and
`time_step`. Preserve time only for splitting and analysis. In working copies,
infinite values become missing. The pipeline fits median imputation, standard
scaling, and L2 logistic regression (`C=1`, `solver=lbfgs`, `class_weight=balanced`,
`max_iter=3000`, `tol=1e-4`, seed 42). Numerical work is restricted to one thread.
A completely missing fitting column stops execution for review.

Balanced weights give equal total loss weight to each class. This is a documented
starting choice for the imbalance; no weighting/parameter search was performed.
Weights can affect probability calibration and the threshold precision/recall tradeoff.

| Set | Time steps | Rows | Class 0 | Class 1 | Illicit share |
| --- | --- | ---: | ---: | ---: | ---: |
| Fit | 1–30 | 26,905 | 23,951 | 2,954 | 10.9794% |
| Validation | 31–35 | 4,330 | 3,640 | 690 | 15.9353% |
| Final refit | 1–35 | 31,235 | 27,591 | 3,644 | 11.6664% |

Imputation/scaling are fitted on fitting-period rows only during validation.
Validation is chronological; no random primary split is used. After evaluation,
a cloned pipeline is refitted on all labelled training rows for test predictions.

| Validation metric | Value |
| --- | ---: |
| ROC-AUC, continuous class-1 probabilities | 0.917141 |
| Precision at threshold 0.5 | 0.342944 |
| Recall at threshold 0.5 | 0.982609 |
| F1 at threshold 0.5 | 0.508436 |

Confusion matrix: rows = actual, columns = predicted, order `[0, 1]`:

```text
[[2341, 1299],
 [  12,  678]]
```

ROC-AUC is threshold-independent. Precision/recall/F1 and the matrix use the
untuned 0.5 threshold. High recall comes with many false positives. Validation
prevalence is higher than fitting prevalence; these metrics are not test-set scores.
Both fits converged (191 iterations for validation, 209 for final fitting).

## Install and run

Verified environment: **Python 3.12.4**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name bt4012-baseline --display-name "BT4012 baseline"
jupyter lab notebooks/01_baseline.ipynb
```

Choose the **BT4012 baseline** kernel, then restart and run all cells. If `python`
already points to Python 3.12, it can be used to create the environment instead.
The kernel is installed within the virtual environment, not globally.

For a clean, headless execution:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=bt4012-baseline \
  --ExecutePreprocessor.timeout=600 notebooks/01_baseline.ipynb
```

To rerun the detailed audit separately:

```bash
python scripts/audit_data.py
```

The notebook prints aggregate checks/results without embedding raw records or
individual predictions. The notebook runs in memory; leave room for the roughly
515 MB of raw CSV data plus processing arrays. Package versions, feature list,
model configuration, split counts, metrics, and raw hashes are recorded for each run.

## Generated outputs

- **`outputs/submission_baseline.csv`**: exactly `index,target`, 15,329 rows.
  `target` contains class-1 probabilities. A one-to-one join to the sample template
  preserves its keys/order. Row count, indices, uniqueness, finite values, missingness,
  and probability bounds are checked before writing and after reading the CSV back.
- **`outputs/baseline_metrics.json`**: aggregate results and reproducibility metadata.
- `reports/data_audit/`: detailed audit tables and reports.

All generated outputs are ignored by Git. Rerunning overwrites the baseline output
files. The final model is reproduced by rerunning the notebook; no fitted-model
binary is committed. No Kaggle submission command is executed.

## GitHub and safety

Use only the existing repository:
**https://github.com/comMinGu/BT4012_Kaggle.git**. Do not create a replacement repo,
force-push, or overwrite an existing remote/history. Preserve the user's private
competition workflow; do not publish to a public repository without explicit approval.

Before committing, review `git status`, `git diff --cached --stat`, and staged paths.
`.gitignore` protects the raw directory, all four raw CSV names anywhere, outputs,
environments, caches, notebook checkpoints, metadata, and common secret files.
It cannot protect secrets manually pasted into code or files already tracked by Git.

The existing repository initially had one README commit. Local `main` is based on
that history, and `origin` points to the URL above. Authenticated Git fetch succeeded.
The repository was initially public and was changed to **private**, then verified,
to honour the requested competition privacy. No existing commits were removed.

GitHub CLI (`gh`) was unavailable in the execution environment. If authentication
is needed, install it locally with `brew install gh`, then run `gh auth login` and
`gh auth setup-git`. Never paste tokens into this notebook or a Git remote URL.

## Next experiment

Compare the same logistic-regression pipeline with `class_weight=None` on the
same fixed temporal split. Compare ROC-AUC and the threshold metrics to understand
how weighting affects ranking and false positives. Keep this run as the reference;
no graph feature engineering or extensive tuning yet. Additional time cutoffs can
later assess how stable the result is across periods.
