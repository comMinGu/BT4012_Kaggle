# Notebooks

`01_baseline.ipynb` is the preserved **E01** Logistic Regression baseline. Its
original local AUC is 0.917141; the user reported public Kaggle AUC 0.88776.
Do not overwrite it when adding experiments.

`02_boosted_tree_experiments.ipynb` evaluates LightGBM and XGBoost with and without
`time_step` using three rolling temporal folds. It also evaluates the unchanged
Logistic Regression specification as a separate E01-R rolling reference. Restart
the Python kernel and run every cell to regenerate the experiment comparisons and
`outputs/submission_tabular_v2.csv`.

`experiments/results.csv` records all configurations consistently. Notebook output
contains aggregate results only; raw records and prediction files stay out of Git.

The full initial audit remains available in `scripts/audit_data.py`. See the root
README for environment setup and command-line execution instructions.
