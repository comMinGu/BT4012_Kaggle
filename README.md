# BT4012 Fraud Analytics — E0906431

The final coursework entry point is **[e0906431.ipynb](e0906431.ipynb)**.
It trains the complete **V22 learned edge-gate ensemble** from the four original
competition files and generates the submission CSV. Its full implementation is
included in [final_model/](final_model/); no older experiment folders, predictions,
pretrained weights or external labels are needed.

## Results

| Result | ROC-AUC | Status |
|---|---:|---|
| V22 best public submission | **0.95716** | Filename identified in the experiment record; public leaderboard verified on 2026-10-25 |
| Final team private leaderboard | **0.94688** | Rank58/80; specific submission filename not independently verified |
| Later V24 submission | 0.95608 public /0.94585 private | Did not improve the public best |

The private team score is not asserted to belong to V22 without a confirmed
submission-to-score mapping. The original V22 CSV SHA256 is
`25dcee6c22846a2937bd00ff9dc118afcb4728ae571423621cebf560e6995d21`.
Fresh raw-data training reproduced the archived V22 ranking and ties exactly;
the maximum score difference was **8.33e-16** (floating-point precision).
[Reproduction evidence](final_model/reproduction_report.json) records the tested
code, environment, temporal diagnostic and comparison with that archived file.

## Setup and run

Use Python **3.12** (tested on3.12.4). Download the competition data with your own
Kaggle access and place these unchanged files in `data/raw/`:

- `train.csv`
- `test.csv`
- `txs_edgelist.csv`
- `sample_submission.csv`

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name bt4012 --display-name "BT4012"
jupyter lab e0906431.ipynb
```

Choose the BT4012 kernel and run all cells. For headless execution:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=bt4012 \
  --ExecutePreprocessor.timeout=7200 e0906431.ipynb
```

Or use the identical implementation from the command line:

```bash
python -m final_model.pipeline --stage all
```

`--stage validate` runs the chronological diagnostic; `--stage final` generates
the test submission only. Both stages use the frozen final recipe. A fresh final
run trains25 models; validation trains another25. CPU training uses at most two
numerical threads. Allow several minutes or longer depending on hardware.

The output is `outputs/v22_reproduction/submission_edgegate_v22_reproduced.csv`,
with15,329 rows, exactly `index,target`, and literal test IDs in sample-template
order. Input hashes, label handling, finite score bounds and CSV round-trip are
checked. Generated checkpoints stay under `outputs/v22_reproduction/`; later
runs can reuse them. If you change model code or settings, use a new output
directory so checkpoints from different recipes cannot be mixed. To force a fresh reproduction:

```bash
python -m final_model.pipeline --stage all --output-dir outputs/v22_fresh
```

The code does not upload to Kaggle. Raw data, trained models and prediction CSVs
are deliberately excluded from Git.

## Model

1. XGBoost400 trees and six CatBoost models (raw336 / filtered244 trees), combined
   with a three-seed,40-epoch bidirectional GraphSAGE ensemble, form V5.
2. An eight-head shared MLP (width128,24 epochs, three seeds) supplies a complementary
   tabular ranking. V9 =75% rank(V5) +25% rank(mean MLP prediction).
3. A three-seed availability-matched quantile GraphSAGE model contributes10% to
   V14, with90% V9 retained.
4. V21 combines95% V14 with5% rank of mean recency-weighted XGBoost margins,
   followed by a10% incoming-neighbor logit correction.
5. Six350-tree depth3 XGBoost classifiers learn direct-edge and shared-parent
   endpoint agreement from known historical labels. Their102 features encode
  32 past-selected quantile features and graph degrees symmetrically.
6. Mean edge probabilities are converted to positive gates. Three anchored
   direct propagation iterations use strength0.6; one sibling correction uses0.2.
   The output is the V22 ranking score, not a calibrated fraud probability.

Feature transformations and selection are fitted only within each past fitting
period. Missing labels are never treated as class0. Graph-only intermediaries
retain topology without invented labels or attributes. Whole-snapshot topology
is an explicit batch-inference assumption, not a real-time availability claim.

## Validation and limitations

The notebook fits through step30 and evaluates known labels in steps31–35, then
fits through step35 and predicts test steps36–49. The diagnostic uses the fixed
final recipe; it is distinct from the original adaptive upstream procedure.
The executed fixed-recipe diagnostic achieved ROC-AUC **0.998650661** on4,330
known-label transactions in steps31–35. It is not a new public result.
Historical windows were repeatedly reused during model development, so neither
the original validation nor this retrospective check is an untouched holdout.
Local gains did not reliably transfer to public leaderboard gains.

The final formula and training lengths are preserved. Neural retraining can vary
slightly across hardware and numerical library versions; inspect the reproduction
report rather than assuming every platform produces byte-identical output.
The original baseline workflow remains in `bt4012.ipynb` for comparison.

## Coursework files

- `e0906431.ipynb`: preprocessing, model explanation, chronological validation,
  complete fitting and CSV generation.
- `final_model/pipeline.py`: raw-data graph construction, fixed orchestration and
  learned edge corrections.
- `final_model/reference_components.py`: tree, tabular neural and full-snapshot
  graph training implementations.
- `final_model/target_graph.py`: availability-matched quantile graph learner.
- `requirements.txt`: pinned dependencies.
- `final_model/reproduction_report.json`: aggregate reproducibility evidence.

Include this repository URL in the separate coursework PDF report. The repository
is currently private; ensure the marker has access. The report itself is not
submitted by running this code or pushing this repository.
