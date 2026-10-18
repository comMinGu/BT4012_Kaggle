# BT4012_Kaggle

Classify Bitcoin transactions as licit (0) or illicit (1), evaluated by ROC-AUC.
All current modelling code is in **[bt4012.ipynb](bt4012.ipynb)**, including data
checks, temporal validation, model comparison, final training, and submission
generation. It does not depend on other notebooks, scripts, or configuration files.

## Files

```text
bt4012.ipynb       Complete modelling and submission workflow
README.md         Setup and execution instructions
requirements.txt  Pinned Python dependencies
```

`.gitignore` keeps raw data, generated predictions, environments, and earlier local
project files out of Git. Earlier versions remain available in Git history.

## Run

Use Python **3.12** (verified on 3.12.4). Download the competition data separately
and place these four unchanged files in `data/raw/`:
`train.csv`, `test.csv`, `sample_submission.csv`, and `txs_edgelist.csv`.
The edge list is checked for integrity but is not used as a model input.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name bt4012 --display-name "BT4012"
jupyter lab bt4012.ipynb
```

Select the **BT4012** kernel, then restart and run all cells. For headless execution:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=bt4012 \
  --ExecutePreprocessor.timeout=900 bt4012.ipynb
```

The notebook creates **`outputs/submission_tabular_v2.csv`**, containing 15,329 rows
with exactly `index,target` in template order. Upload that local CSV to Kaggle
manually. Raw data and generated predictions are not hosted in this repository.

## Current approach and results

Only labelled rows are used for fitting. Compare Logistic Regression, LightGBM,
and XGBoost across three temporal folds: fit through 20/25/30, validate on
21–25/26–30/31–35. LightGBM and XGBoost are each compared with and without
`time_step`. Fixed settings, fitting-only class weights, seed 42, raw hashes,
selection criteria, and all fold results are documented inside the notebook.
No graph features, pseudo-labels, or parameter search are used.

The selected model is **E04: XGBoost with the 165 anonymized features**. Its fold
AUCs are **0.940656 / 0.995593 / 0.997495**, mean **0.977915**, and population
standard deviation **0.026357**. The original Logistic Regression had latest-fold
AUC **0.917141** and user-reported public Kaggle AUC **0.88776**. E04's public score
is unknown. Local validation also selects the model and does not predict future
leaderboard performance; the first fold remains much harder.

Before final coursework submission, rename `bt4012.ipynb` to your NUS student
number, for example `e1234567.ipynb`, and update its filename in these instructions.
Include the repository link in your report and ensure the marker can access this
private repository.
