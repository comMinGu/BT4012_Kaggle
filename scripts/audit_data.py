#!/usr/bin/env python3
"""Read-only dataset audit. Run from any directory; outputs contain no model fit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"feat_{i}" for i in range(1, 166)]
PURPOSES = {
    "train.csv": "Transaction features and partially observed training labels",
    "test.csv": "Future-period transaction features and submission row keys; no labels",
    "txs_edgelist.csv": "Directed transaction connections: txId1 -> txId2",
    "sample_submission.csv": "Submission template: index key and placeholder target score",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown tables without an optional tabulate dependency."""
    rows = [list(map(str, frame.columns)), ["---"] * len(frame.columns)]
    rows.extend([[str(value) for value in row] for row in frame.itertuples(index=False, name=None)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def profile(name: str, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    columns = []
    for column in frame:
        values = frame[column]
        numeric = pd.api.types.is_numeric_dtype(values)
        columns.append({
            "file": name,
            "column": column,
            "dtype": str(values.dtype),
            "missing": int(values.isna().sum()),
            "missing_percent": float(values.isna().mean() * 100),
            "unique_nonmissing": int(values.nunique()),
            "constant_nonmissing": bool(values.nunique() == 1),
            "positive_infinity": int(np.isposinf(values).sum()) if numeric else 0,
            "negative_infinity": int(np.isneginf(values).sum()) if numeric else 0,
            "min": values.min() if numeric else None,
            "max": values.max() if numeric else None,
        })
    ids = {}
    for column in ["txId", "index", "txId1", "txId2"]:
        if column in frame:
            values = frame[column]
            ids[column] = {
                "unique": int(values.nunique()),
                "missing": int(values.isna().sum()),
                "repeated_occurrences_after_first": int(values.dropna().duplicated().sum()),
                "rows_in_repeated_id_groups": int(values.notna().mul(values.duplicated(keep=False)).sum()),
            }
    result = {
        "purpose": PURPOSES.get(name, "Undocumented additional CSV"),
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows_after_first": int(frame.duplicated().sum()),
        "rows_in_duplicate_groups": int(frame.duplicated(keep=False).sum()),
        "id_checks": ids,
    }
    return result, pd.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw",
                        help="Defaults to the repository's data/raw directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "data_audit")
    args = parser.parse_args()
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    require(raw.is_dir(), f"Raw directory not found: {raw}")
    require(output != raw and raw not in output.parents, "Outputs must be outside the raw directory")
    paths = sorted(p for p in raw.rglob("*") if p.is_file())
    require(bool(paths), f"No raw files found in {raw}")
    inventory = {
        str(p.relative_to(raw)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in paths
    }
    frames, profiles, column_profiles = {}, {}, []
    for path in paths:
        name = str(path.relative_to(raw))
        print(f"Inspecting {name}", flush=True)
        if path.suffix.lower() != ".csv":
            inventory[name]["inspection"] = "Non-CSV: inventory/hash only; manual inspection required"
            continue
        # Reject malformed widths and duplicate headers instead of letting the parser hide them.
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            require(len(header) == len(set(header)), f"Duplicate column names in {name}")
            for line, row in enumerate(reader, start=2):
                require(len(row) == len(header), f"Wrong field count: {name}, record {line}")
        frame = pd.read_csv(path, low_memory=False)
        frames[name] = frame
        profiles[name], column_profile = profile(name, frame)
        column_profiles.append(column_profile)
        inventory[name]["inspection"] = "All records parsed and profiled"
    require(set(PURPOSES).issubset(frames), "One or more required competition CSV files are missing")
    train, test, edges, sample = [frames[n] for n in PURPOSES]
    expected = {
        "train.csv": ["txId", "time_step", *FEATURES, "label"],
        "test.csv": ["index", "txId", "time_step", *FEATURES],
        "txs_edgelist.csv": ["txId1", "txId2"],
        "sample_submission.csv": ["index", "target"],
    }
    for name, columns in expected.items():
        require(frames[name].columns.tolist() == columns, f"Schema changed in {name}; inspect before adapting")
    require(set(train.label.dropna().unique()).issubset({0, 1}), "Unexpected label encoding")
    for name, columns in {
        "train.csv": ["txId", "time_step"], "test.csv": ["index", "txId", "time_step"],
        "txs_edgelist.csv": ["txId1", "txId2"], "sample_submission.csv": ["index"],
    }.items():
        for column in columns:
            require(pd.api.types.is_integer_dtype(frames[name][column]), f"Expected complete integer keys: {name}.{column}")
    for name, column in [("train.csv", "txId"), ("test.csv", "txId"), ("test.csv", "index"), ("sample_submission.csv", "index")]:
        require(frames[name][column].is_unique, f"Duplicate keys: {name}.{column}; resolve join ambiguity")

    train_ids, test_ids = set(train.txId), set(test.txId)
    feature_ids = train_ids | test_ids
    edge_ids = set(edges.txId1) | set(edges.txId2)
    require(not train_ids & test_ids, "Train/test transaction IDs overlap; resolve join ambiguity")
    all_ids = sorted(feature_ids | edge_ids)
    supplied = pd.concat([
        train[["txId", "time_step", "label"]].assign(split="train"),
        test[["txId", "time_step"]].assign(label=np.nan, split="test"),
    ], ignore_index=True).set_index("txId")
    nodes = supplied.reindex(pd.Index(all_ids, name="txId"))
    nodes["split"] = nodes["split"].fillna("edge_only")
    nodes["has_features"] = nodes.index.isin(feature_ids)
    nodes["has_label"] = nodes.label.notna()
    # Count every supplied directed edge row. Duplicates/loops are reported, not removed.
    nodes["in_degree"] = edges.txId2.value_counts().reindex(nodes.index, fill_value=0).astype("int64")
    nodes["out_degree"] = edges.txId1.value_counts().reindex(nodes.index, fill_value=0).astype("int64")
    nodes["total_degree"] = nodes.in_degree + nodes.out_degree
    nodes["isolated"] = nodes.total_degree.eq(0)
    require(int(nodes.in_degree.sum()) == len(edges), "In-degree sum must equal edge count")
    require(int(nodes.out_degree.sum()) == len(edges), "Out-degree sum must equal edge count")

    edge_audit = edges.copy()
    for endpoint, column in [("source", "txId1"), ("destination", "txId2")]:
        edge_audit[f"{endpoint}_split"] = edges[column].map(nodes.split)
        edge_audit[f"{endpoint}_time_step"] = edges[column].map(nodes.time_step)
    source_time = edge_audit.source_time_step
    destination_time = edge_audit.destination_time_step
    known_time = source_time.notna() & destination_time.notna()
    memberships = edge_audit.groupby(["source_split", "destination_split"]).size().rename("edges").reset_index()
    graph = {
        "feature_nodes": len(feature_ids),
        "edge_endpoint_nodes": len(edge_ids),
        "all_nodes_union": len(all_ids),
        "edge_only_nodes_without_features_or_labels": len(edge_ids - feature_ids),
        "directed_edge_rows": len(edges),
        "unique_directed_edges": len(edges.drop_duplicates()),
        "duplicate_edge_rows_after_first": int(edges.duplicated().sum()),
        "self_loops": int(edges.txId1.eq(edges.txId2).sum()),
        "isolated_feature_nodes": int(nodes.loc[nodes.has_features, "isolated"].sum()),
        "isolated_nodes_all_union": int(nodes.isolated.sum()),
        "edges_both_endpoints_have_features": int((edges.txId1.isin(feature_ids) & edges.txId2.isin(feature_ids)).sum()),
        "source_rows_without_features": int((~edges.txId1.isin(feature_ids)).sum()),
        "destination_rows_without_features": int((~edges.txId2.isin(feature_ids)).sum()),
        "edges_same_known_time_step": int((known_time & source_time.eq(destination_time)).sum()),
        "edges_forward_known_time_step": int((known_time & source_time.lt(destination_time)).sum()),
        "edges_backward_known_time_step": int((known_time & source_time.gt(destination_time)).sum()),
        "edges_with_unknown_endpoint_time": int((~known_time).sum()),
        "edge_split_counts": memberships.to_dict(orient="records"),
    }
    degree_summaries, degree_frequencies = [], []
    for universe, subset in [("all_nodes_union", nodes), ("feature_nodes", nodes[nodes.has_features]),
                             ("train", nodes[nodes.split.eq("train")]), ("test", nodes[nodes.split.eq("test")]),
                             ("edge_only", nodes[nodes.split.eq("edge_only")])]:
        for direction in ["in_degree", "out_degree"]:
            values = subset[direction]
            stats = values.describe(percentiles=[.25, .5, .75, .9, .95, .99]).to_dict()
            degree_summaries.append({"universe": universe, "direction": direction, **stats,
                                     "zero_degree_nodes": int(values.eq(0).sum())})
            freq = values.value_counts().sort_index().rename_axis("degree").reset_index(name="nodes")
            freq.insert(0, "direction", direction)
            freq.insert(0, "universe", universe)
            degree_frequencies.append(freq)

    label_counts = train.label.value_counts().sort_index()
    labels = {
        "labelled_train": int(train.label.notna().sum()),
        "unlabelled_train": int(train.label.isna().sum()),
        "test_without_label_column": len(test),
        "counts": {str(int(k)): int(v) for k, v in label_counts.items()},
        "percent_of_labelled": {str(int(k)): float(v / label_counts.sum() * 100) for k, v in label_counts.items()},
        "numeric_class_mapping": {"0": "negative", "1": "positive / illicit"},
        "class_mapping_source": "Explicit user instruction; semantics are not documented in the CSVs",
    }
    temporal = supplied.reset_index()
    temporal["label_status"] = np.where(temporal.split.eq("test"), "test_hidden",
                                         temporal.label.map({0: "label_0", 1: "label_1"}).fillna("train_unlabelled"))
    time_counts = temporal.groupby(["split", "time_step", "label_status"]).size().rename("rows").reset_index()
    time_ranges = temporal.groupby(["split", "label_status"]).agg(
        rows=("txId", "size"), min_time_step=("time_step", "min"),
        max_time_step=("time_step", "max"), distinct_time_steps=("time_step", "nunique"),
    ).reset_index()
    labelled = train.loc[train.label.notna()]
    baseline_split = []
    for split, first_step, last_step in [("training", 1, 30), ("validation", 31, 35)]:
        subset = labelled.loc[labelled.time_step.between(first_step, last_step)]
        baseline_split.append({
            "split": split, "first_time_step": first_step, "last_time_step": last_step,
            "rows": len(subset), "class_0": int(subset.label.eq(0).sum()),
            "class_1": int(subset.label.eq(1).sum()),
            "class_1_percent": round(float(subset.label.mean() * 100), 4),
        })

    # Direct value comparisons, excluding IDs/labels; no feature deduplication is performed.
    feature_rows = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    with_time = pd.concat([train[["time_step", *FEATURES]], test[["time_step", *FEATURES]]], ignore_index=True)
    shared_patterns = train[FEATURES].drop_duplicates().merge(test[FEATURES].drop_duplicates(), how="inner", on=FEATURES)
    duplicate_features = train.loc[train[FEATURES].duplicated(keep=False)].copy()
    duplicate_features["feature_pattern"] = duplicate_features.groupby(FEATURES, sort=True, dropna=False).ngroup()
    duplicate_label_counts = duplicate_features.groupby("feature_pattern").label.nunique()
    alignment = {
        "train_test_txId_overlap": len(train_ids & test_ids),
        "test_and_submission_index_sets_match": set(test["index"]) == set(sample["index"]),
        "test_and_submission_index_order_matches": test["index"].equals(sample["index"]),
        "test_index_is_zero_based_row_sequence": bool(np.array_equal(test["index"], np.arange(len(test)))),
        "train_test_common_predictor_columns_match": train.columns.drop("label").tolist() == test.columns.drop("index").tolist(),
        "strictly_later_test_time_steps": bool(train.time_step.max() < test.time_step.min()),
        "overlapping_time_steps": sorted(set(train.time_step) & set(test.time_step)),
        "feature_only_duplicate_rows_train": int(train[FEATURES].duplicated().sum()),
        "feature_only_duplicate_rows_test": int(test[FEATURES].duplicated().sum()),
        "feature_only_duplicate_rows_combined": int(feature_rows.duplicated().sum()),
        "feature_and_time_duplicate_rows_combined": int(with_time.duplicated().sum()),
        "distinct_feature_patterns_shared_train_test": len(shared_patterns),
        "train_rows_in_duplicate_feature_groups": len(duplicate_features),
        "train_duplicate_feature_groups": int(duplicate_label_counts.size),
        "train_duplicate_feature_groups_with_conflicting_known_labels": int(duplicate_label_counts.gt(1).sum()),
        "submission_target_values": sorted(sample.target.unique().tolist()),
    }
    require(alignment["test_and_submission_index_sets_match"], "Submission keys do not match test keys")
    unchanged = all(sha256(raw / name) == entry["sha256"] for name, entry in inventory.items())
    require(unchanged, "Raw input content changed during audit")
    summary = {
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "raw_directory_name": raw.name,
        "raw_hashes_unchanged_after_audit": unchanged,
        "inventory": inventory, "files": profiles, "labels": labels,
        "alignment": alignment, "graph": graph, "baseline_temporal_split": baseline_split,
    }
    output.mkdir(parents=True, exist_ok=True)
    column_frame = pd.concat(column_profiles, ignore_index=True)
    column_frame.to_csv(output / "column_profile.csv", index=False)
    nodes.to_csv(output / "node_audit.csv", index=True)
    nodes[nodes.isolated].to_csv(output / "isolated_nodes.csv", index=True)
    nodes[~nodes.has_features].to_csv(output / "edge_only_nodes.csv", index=True)
    edge_audit.to_csv(output / "edge_audit.csv", index=False)
    memberships.to_csv(output / "edge_split_counts.csv", index=False)
    pd.DataFrame(degree_summaries).to_csv(output / "degree_summary.csv", index=False)
    pd.concat(degree_frequencies, ignore_index=True).to_csv(output / "degree_distribution.csv", index=False)
    time_counts.to_csv(output / "time_label_counts.csv", index=False)
    time_ranges.to_csv(output / "time_ranges.csv", index=False)
    pd.DataFrame(baseline_split).to_csv(output / "baseline_temporal_split.csv", index=False)
    duplicate_features[["feature_pattern", "txId", "time_step", "label"]].to_csv(
        output / "duplicate_feature_rows.csv", index=False
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    file_table = pd.DataFrame([
        {"File": name, "Rows": p["rows"], "Columns": p["columns"],
         "Missing cells": sum(p["missing_by_column"].values()), "Duplicate rows": p["duplicate_rows_after_first"]}
        for name, p in profiles.items()
    ])
    nonfinite = int(column_frame.positive_infinity.sum() + column_frame.negative_infinity.sum())
    degree_table = pd.DataFrame(degree_summaries)
    degree_table = degree_table[degree_table.universe.eq("all_nodes_union")].drop(columns="universe").round(4)
    report = f"""# Dataset audit

Generated by `python scripts/audit_data.py`. Counts come from the supplied CSVs.
Metric: ROC-AUC (provided by the user). This audit does not fit a model or clean data.

## File inventory

{markdown_table(file_table)}

All raw file SHA-256 hashes were unchanged after the audit. Exact hashes and the
runtime versions are in `summary.json`. All CSV records were checked for field counts.

## Schema and conventions

- `train.csv`: `txId`, `time_step`, `feat_1` through `feat_165`, `label`.
- `test.csv`: `index`, `txId`, `time_step`, `feat_1` through `feat_165`.
- `txs_edgelist.csv`: `txId1`, `txId2`.
- `sample_submission.csv`: `index`, `target`.
- `txId`, endpoint IDs, `index`, and `time_step` parse as `int64`.
  Features, the nullable training `label`, and template `target` parse as `float64`.
  The label dtype reflects blank cells; its observed nonmissing values are binary.
- Every column's exact name, inferred dtype, missing count, unique count,
  numeric range, constant flag, and infinity counts are in `column_profile.csv`.
- CSVs have no embedded feature definitions. Do not infer what any `feat_*`
  measures, how it was normalised, or which features aggregate graph neighbours.
- `time_step` is an integer ordering variable. Calendar dates and interval duration
  are not documented in these files.
- Missingness uses pandas' default CSV NA tokens, including blank cells. No values
  were imputed, scaled, relabelled, or removed. Numeric infinities found: {nonfinite}.

## Labels and temporal split

- Labelled training rows: {labels['labelled_train']:,}; unlabelled training rows:
  {labels['unlabelled_train']:,}. Use `label.notna()` for supervised training.
- Class 0: {labels['counts'].get('0', 0):,}
  ({labels['percent_of_labelled'].get('0', 0):.4f}% of labelled rows).
- Class 1: {labels['counts'].get('1', 0):,}
  ({labels['percent_of_labelled'].get('1', 0):.4f}% of labelled rows).
- Test has {len(test):,} rows and no label column. Blank training labels are
  unknown observations, not class 0 and not submission rows.
- The user explicitly defines class 0 as negative and class 1 as positive / illicit.
  This mapping comes from the user, not undocumented CSV semantics. Submission
  scores must represent the probability of class 1.
- The modelling target is training `label`; submission `target` is a score column.
  All observed template scores are {alignment['submission_target_values']}, placeholders rather than labels.

{markdown_table(time_ranges)}

Exact time-step/class counts are in `time_label_counts.csv`. The observed split is
strictly chronological: {alignment['strictly_later_test_time_steps']}.
The files do not document how test transactions were selected from the later period.

The requested baseline fits labelled rows through time step 30 and validates on
labelled rows from steps 31–35. Both sets contain both classes:

{markdown_table(pd.DataFrame(baseline_split))}

## Cross-file identity checks

- Train/test transaction ID overlap: {alignment['train_test_txId_overlap']}.
- Transaction keys and submission keys passed uniqueness and nonmissing checks.
- Test/submission `index` sets match: {alignment['test_and_submission_index_sets_match']};
  order matches: {alignment['test_and_submission_index_order_matches']}.
- Test `index` is a zero-based row sequence: {alignment['test_index_is_zero_based_row_sequence']}.
  Map predictions using `test["index"]` to `sample_submission["index"]`;
  join graph information using `txId`. These keys have different purposes.
- Feature columns/order match between train and test:
  {alignment['train_test_common_predictor_columns_match']}.
- Duplicate feature vectors (excluding IDs, time, and labels): train
  {alignment['feature_only_duplicate_rows_train']}, test
  {alignment['feature_only_duplicate_rows_test']}, combined
  {alignment['feature_only_duplicate_rows_combined']}.
  Distinct feature patterns shared across train/test:
  {alignment['distinct_feature_patterns_shared_train_test']}.
  Combined duplicate vectors including time: {alignment['feature_and_time_duplicate_rows_combined']}.
- Training rows participating in duplicate-feature groups:
  {alignment['train_rows_in_duplicate_feature_groups']} across {alignment['train_duplicate_feature_groups']} groups.
  Groups with conflicting known labels: {alignment['train_duplicate_feature_groups_with_conflicting_known_labels']}.
  `duplicate_feature_rows.csv` lists the affected transaction IDs and label/time
  context. These are distinct transactions with matching features; none was removed.

## Directed graph and endpoint coverage

Treat each row as `txId1 -> txId2`, consistent with the supplied description of
payment flows. The header alone does not independently prove payment direction.
There are no edge weights, timestamps, or payment amounts in the edge file.
Endpoint IDs refer to transaction IDs, never DataFrame row positions or submission indices.

- Nodes with supplied features: {graph['feature_nodes']:,}.
- Distinct edge endpoints: {graph['edge_endpoint_nodes']:,}.
- Node universe (feature IDs union edge IDs): {graph['all_nodes_union']:,}.
- Edge-only nodes with no supplied feature row/label/time: {graph['edge_only_nodes_without_features_or_labels']:,}.
- Directed edge rows: {graph['directed_edge_rows']:,}; unique directed edges:
  {graph['unique_directed_edges']:,}; repeated edge rows: {graph['duplicate_edge_rows_after_first']:,};
  self-loops: {graph['self_loops']:,}.
- Isolated feature nodes: {graph['isolated_feature_nodes']:,} (no incident edges in
  the supplied full edge list). Isolated nodes in the union: {graph['isolated_nodes_all_union']:,}.
- Edges with both endpoints in feature tables: {graph['edges_both_endpoints_have_features']:,}.
- Source occurrences without features: {graph['source_rows_without_features']:,};
  destination occurrences without features: {graph['destination_rows_without_features']:,}.

{markdown_table(memberships)}

The edge-only IDs are a coverage gap, not evidence of malformed IDs. Their source,
labels, and time steps cannot be recovered from these CSVs. Keep them distinct from
unlabelled training rows and test rows. `edge_only_nodes.csv` lists them explicitly.

### Degree distributions

In-degree counts incoming edge rows; out-degree counts outgoing rows. The following
statistics use the full node union, include zero degrees, and use all supplied edges.
Standard deviation is the sample standard deviation; quantiles use pandas defaults.
Both degree sums equal the number of edges (checked).

{markdown_table(degree_table)}

`degree_distribution.csv` contains exact frequency tables, including degree zero;
`degree_summary.csv` also reports feature-only, train, test, and edge-only subsets.
These subset statistics still count edges in the full supplied graph; they are not
degrees in an induced subgraph. `node_audit.csv` lists each node's group, label
availability, known time, degrees, and isolation status.

### Edge/time alignment

- Same known time step: {graph['edges_same_known_time_step']:,} edges.
- Later destination time: {graph['edges_forward_known_time_step']:,} edges.
- Earlier destination time: {graph['edges_backward_known_time_step']:,} edges.
- At least one unknown endpoint time: {graph['edges_with_unknown_endpoint_time']:,} edges.

Time comparisons apply only when both endpoint feature rows are available. Even
within one time step, transaction order and feature availability are undocumented.

## Leakage risks and next step

1. Use an earlier-to-later validation split on labelled training rows, because the
   competition test lies in later time steps. A random split can mask time drift.
   Keep both classes in validation and report their counts alongside ROC-AUC.
2. Never fill unknown training labels with zero or use test/template scores as labels.
   Do not calculate neighbour-label features using validation/test outcomes.
3. Use only `feat_1` through `feat_165` for the first baseline. Exclude `txId`,
   submission `index`, and `label`. Preserve `time_step` for the temporal split and
   analysis, but exclude it from model features.
4. Fit imputation, scaling, selection, resampling, and tuning only on each training
   fold. The supplied feature construction/normalisation history is unknown.
5. Global graph features can incorporate future structure. Graph-only nodes have no
   known time; even known same-step edges lack within-step chronology. Define an
   availability policy and confirm competition rules before using full-graph features.
6. Baseline design: a fixed temporal holdout (steps 1–30 to fit, 31–35 to validate),
   with a simple feature-only logistic-regression pipeline and a constant-score sanity
   reference. Start with the 165 supplied features, fixed seeds, train-fitted scaling,
   and ROC-AUC on continuous class-1 probabilities. Use labelled observations only.
   Model fitting and submission generation belong to `notebooks/01_baseline.ipynb`;
   this audit performs no modelling or graph feature engineering.

## Reproduce

From the repository root, install `requirements.txt` into a Python 3.12 virtual
environment and run `python scripts/audit_data.py`. The raw files live in `data/raw/`,
the script's default input directory. Their bytes were preserved when relocated.
Use `--raw-dir /path/to/raw` and `--output-dir /path/to/output` to override.
Generated row-level outputs are ignored by Git. `docs/data_audit.md` is a versionable
snapshot of this summary; refresh it intentionally after reviewing a rerun.
"""
    (output / "data_audit.md").write_text(report)
    print(f"Audit complete: {output / 'data_audit.md'}", flush=True)


if __name__ == "__main__":
    main()
