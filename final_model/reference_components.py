# Frozen reference components copied from E31; runtime globals are set by the driver.
from pathlib import Path

import copy,gc,hashlib,json,math,time

import numpy as np

import pandas as pd

import scipy.sparse as sp

from scipy.stats import rankdata

from scipy.special import logit

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import QuantileTransformer,StandardScaler

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier,early_stopping,log_evaluation

from catboost import CatBoostClassifier

from threadpoolctl import threadpool_limits

import joblib,psutil,torch

from torch import nn

from torch.nn import functional as F

def save_json(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def save(n,x):save_json(CACHE/n,x)

def rank(x):return (rankdata(x,method='average')-.5)/len(x)

TREE_PROFILES={
 'P01':dict(family='XGB',depth=5,learning_rate=.05,child=5,reg_lambda=1.,reg_alpha=0.,colsample=1.,subsample=.8,power=1.,max_rounds=800),
 'P06':dict(family='CAT',depth=4,learning_rate=.06,reg_lambda=8.,power=.5,max_rounds=600,boosting='Ordered')}

class TreeRunner:
    """Fit known labels only, caching column selections but never data copies.

    X64 preserves original float64 inputs for the exact E04 and CatBoost
    references. New profiles use float32 feature matrices to bound memory.
    Group B uses original features, S adds graph structure, and N additionally
    adds incoming/outgoing means. The drop policy removes original columns with
    absolute Spearman correlation >=0.90 with time in the fitting labels only;
    the corresponding neighbor mean columns are removed as well. Graph structure
    stays intact, and time itself is never included as a predictor.

    New configurations use four numerical threads. E04 uses exactly one, while
    the CatBoost reference uses two to reproduce the existing reference setup.
    """

    def __init__(self, X64, structure, incoming_mean, outgoing_mean, labels, times, cache_dir):
        self.X64 = np.asarray(X64)
        if self.X64.dtype != np.float64:
            raise ValueError("X64 must preserve the original float64 feature matrix.")
        self.structure = np.asarray(structure)
        self.incoming_mean = np.asarray(incoming_mean)
        self.outgoing_mean = np.asarray(outgoing_mean)
        self.y = np.asarray(labels, dtype=np.float64)
        self.times = np.asarray(times)
        n = len(self.X64)
        if self.X64.ndim != 2 or self.structure.ndim != 2:
            raise ValueError("Original and graph structure features must be matrices.")
        if len(self.structure) != n or self.incoming_mean.shape != self.X64.shape or self.outgoing_mean.shape != self.X64.shape:
            raise ValueError("All feature blocks must share the supplied feature-row order.")
        if self.y.shape != (n,) or self.times.shape != (n,):
            raise ValueError("Labels and times must align with feature rows.")
        if not np.isfinite(self.times).all():
            raise ValueError("Every supplied row requires a finite time step.")
        self.known = np.isfinite(self.y)
        if not np.isin(self.y[self.known], [0, 1]).all():
            raise ValueError("Known targets must be class 0 or 1.")
        if np.isinf(self.y).any():
            raise ValueError("Unknown labels must be NaN, never infinity.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fit_log = []
        self._column_cache = {}

    def _row_indices(self, values, name):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer index array.")
        if len(values) == 0 or (values < 0).any() or (values >= len(self.y)).any():
            raise ValueError(f"{name} is empty or contains out-of-range indices.")
        return values.astype(np.int64, copy=False)

    def _columns(self, cutoff, policy, fit_idx):
        key = (int(cutoff), policy)
        if key not in self._column_cache:
            if policy == "raw":
                removed = np.empty(0, dtype=np.int64)
            elif policy == "drop":
                base = self.X64[fit_idx].copy()
                base[~np.isfinite(base)] = np.nan
                correlation = pd.DataFrame(base).corrwith(
                    pd.Series(self.times[fit_idx]), method="spearman"
                ).fillna(0).abs().to_numpy()
                removed = np.flatnonzero(correlation >= .90)
            else:
                raise ValueError("Feature policy must be raw or drop.")
            kept = np.setdiff1d(np.arange(self.X64.shape[1]), removed, assume_unique=True)
            if len(kept) == 0:
                raise ValueError("Time-proxy removal would remove every original feature.")
            self._column_cache[key] = (kept, removed)
        return self._column_cache[key]

    def _matrix(self, rows, columns, group, dtype):
        blocks = [self.X64[np.ix_(rows, columns)]]
        if group in {"S", "N"}:
            blocks.append(self.structure[rows])
        if group == "N":
            blocks.extend([
                self.incoming_mean[np.ix_(rows, columns)],
                self.outgoing_mean[np.ix_(rows, columns)],
            ])
        if group not in {"B", "S", "N"}:
            raise ValueError("Feature group must be B, S, or N.")
        matrix = np.concatenate(blocks, axis=1).astype(dtype, copy=False)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    def fit(self, config, cutoff, pred_idx, seed=42, rounds=None, early_idx=None, stage="development"):
        """Return class-1 probabilities, used rounds, and aggregate fit metadata.

        When rounds is None, supply labelled early_idx strictly later than the
        fitting cutoff. That inner validation controls early stopping only.
        Fixed-round fits never receive an evaluation set. pred_idx is separate
        from early_idx and its labels are not inspected unless explicitly reused
        as early_idx by the caller. The caller must reserve early_idx inside each
        outer fitting period, never its outer evaluation window.
        """
        started = time.perf_counter()
        cutoff = int(cutoff)
        fit_idx = np.flatnonzero(self.known & (self.times <= cutoff))
        if len(fit_idx) == 0 or len(np.unique(self.y[fit_idx])) != 2:
            raise ValueError("Fitting rows must contain known examples of both classes.")
        pred_idx = self._row_indices(pred_idx, "pred_idx")
        if np.any(self.times[pred_idx] <= cutoff):
            raise ValueError("Prediction rows must occur after the fitting cutoff.")
        reference = config.get("reference")
        group = config.get("group", "B")
        policy = config.get("policy", "raw")
        if reference is not None and group != "B":
            raise ValueError("Historical reference models use original features only.")
        if reference == "E04":
            if policy != "raw" or (rounds is not None and int(rounds) != 400) or early_idx is not None:
                raise ValueError("E04 requires raw original features and exactly 400 fixed rounds.")
            profile = dict(TREE_PROFILES["P01"], max_rounds=400)
            rounds = 400
            threads = 1
        elif reference == "CAT":
            profile = dict(TREE_PROFILES["P06"], max_rounds=700)
            threads = 2
        elif reference is None:
            if config.get("profile") not in TREE_PROFILES:
                raise ValueError("Unknown prespecified tree profile.")
            profile = TREE_PROFILES[config["profile"]]
            threads = 4
        else:
            raise ValueError("Unknown historical reference type.")
        early = rounds is None
        if early:
            if early_idx is None:
                raise ValueError("Early-stopping rows are required when rounds is None.")
            early_idx = self._row_indices(early_idx, "early_idx")
            if not self.known[early_idx].all() or np.any(self.times[early_idx] <= cutoff):
                raise ValueError("Early-stopping rows must be known labels strictly after the fitting cutoff.")
            if len(np.unique(self.y[early_idx])) != 2:
                raise ValueError("Early-stopping rows must contain both target classes.")
            cap = int(profile["max_rounds"])
        else:
            if early_idx is not None:
                raise ValueError("Fixed-round fitting cannot receive early-stopping rows.")
            if int(rounds) != rounds or int(rounds) < 1:
                raise ValueError("Boosting rounds must be a positive integer.")
            cap = int(rounds)
        columns, removed = self._columns(cutoff, policy, fit_idx)
        dtype = np.float64 if reference is not None else np.float32
        a = self._matrix(fit_idx, columns, group, dtype)
        b = self._matrix(pred_idx, columns, group, dtype)
        e = self._matrix(early_idx, columns, group, dtype) if early else None
        fit_y = self.y[fit_idx].astype(np.int64)
        negative = int((fit_y == 0).sum())
        positive = int((fit_y == 1).sum())
        ratio = float((negative / positive) ** profile["power"])
        family = profile["family"]
        if family == "XGB":
            params = dict(
                n_estimators=cap, learning_rate=profile["learning_rate"], max_depth=profile["depth"],
                min_child_weight=profile["child"], subsample=profile["subsample"],
                colsample_bytree=profile["colsample"], reg_lambda=profile["reg_lambda"],
                reg_alpha=profile["reg_alpha"], objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device="cpu", random_state=int(seed), n_jobs=threads,
                scale_pos_weight=ratio,
            )
            if early:
                params["early_stopping_rounds"] = 80
            model = XGBClassifier(**params)
            with threadpool_limits(limits=threads):
                model.fit(a, fit_y, sample_weight=np.ones(len(fit_idx)) if reference == "E04" else None,
                          **({"eval_set": [(e, self.y[early_idx])], "verbose": False} if early else {}))
                p = model.predict_proba(b)[:, list(model.classes_).index(1)]
            used = int(model.best_iteration + 1) if early else cap
        elif family == "LGB":
            model = LGBMClassifier(
                n_estimators=cap, learning_rate=profile["learning_rate"], num_leaves=profile["leaves"],
                max_depth=profile["depth"], min_child_samples=profile["child"],
                subsample=profile["subsample"], subsample_freq=1,
                colsample_bytree=profile["colsample"], reg_lambda=profile["reg_lambda"], reg_alpha=0.,
                scale_pos_weight=ratio, objective="binary", metric="auc", random_state=int(seed), n_jobs=threads,
                deterministic=True, force_col_wise=True, verbosity=-1,
            )
            with threadpool_limits(limits=threads):
                model.fit(a, fit_y, **({
                    "eval_set": [(e, self.y[early_idx])], "eval_metric": "auc",
                    "callbacks": [early_stopping(80, first_metric_only=True, verbose=False), log_evaluation(0)],
                } if early else {}))
                p = model.predict_proba(b)[:, list(model.classes_).index(1)]
            used = int(model.best_iteration_) if early else cap
        else:
            model = CatBoostClassifier(
                iterations=cap, depth=profile["depth"], learning_rate=profile["learning_rate"],
                l2_leaf_reg=profile["reg_lambda"], random_strength=.5,
                bootstrap_type="Bayesian", bagging_temperature=.5, boosting_type="Ordered",
                loss_function="Logloss", eval_metric="AUC:use_weights=false", class_weights=[1., ratio],
                random_seed=int(seed), thread_count=threads, verbose=False,
                allow_writing_files=False, task_type="CPU", border_count=128,
            )
            with threadpool_limits(limits=threads):
                model.fit(a, fit_y, sample_weight=np.ones(len(fit_idx)) if reference == "CAT" else None, **({
                    "eval_set": (e, self.y[early_idx]), "early_stopping_rounds": 80, "use_best_model": True,
                } if early else {"use_best_model": False}))
                p = model.predict_proba(b)[:, list(model.classes_).index(1)]
            used = int(model.tree_count_)
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (len(pred_idx),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError("Invalid predicted class-1 probabilities.")
        info = {
            "stage": stage, "config": config.get("id", reference), "family": family,
            "reference": reference or "", "group": group, "policy": policy,
            "profile": config.get("profile", reference), "fit_end": cutoff,
            "fit_rows": len(fit_idx), "fit_negative": negative, "fit_positive": positive,
            "seed": int(seed), "rounds": used, "round_cap": cap, "early_stopping": early,
            "early_min_time": int(self.times[early_idx].min()) if early else None,
            "early_max_time": int(self.times[early_idx].max()) if early else None,
            "n_features": a.shape[1], "removed": len(removed),
            "removed_features": [f"feat_{j + 1}" for j in removed],
            "class_weight_power": profile["power"], "positive_multiplier": ratio,
            "threads": threads, "matrix_dtype": str(a.dtype),
            "rss_gib": psutil.Process().memory_info().rss / 2**30,
            "seconds": round(time.perf_counter() - started, 6),
        }
        self.fit_log.append(info)
        log = pd.DataFrame(self.fit_log)
        log["removed_features"] = log["removed_features"].map(json.dumps)
        log.to_csv(self.cache_dir / "tree_fit_log.csv", index=False)
        joblib.dump(model, CACHE/f"{MODEL_TAG}_model.joblib")
        np.save(CACHE/f"{MODEL_TAG}_columns.npy",columns)
        del model, a, b, e
        return p, used, info

class BidirectionalSageLayer(nn.Module):
    """Keep incoming, outgoing, and own-node information in distinct channels."""

    def __init__(self, input_width, output_width, message_dropout=0.0):
        super().__init__()
        self.own = nn.Linear(input_width, output_width)
        self.incoming = nn.Linear(input_width, output_width, bias=False)
        self.outgoing = nn.Linear(input_width, output_width, bias=False)
        self.norm = nn.LayerNorm(output_width)
        self.message_dropout = float(message_dropout)

    def forward(self, values, incoming, outgoing):
        messages = values
        if self.training and self.message_dropout > 0:
            # Label-independent node dropout affects message channels only.
            # The focal node's own projection retains its complete input.
            keep = torch.rand((len(values), 1), device=values.device) >= self.message_dropout
            messages = values * keep / (1.0 - self.message_dropout)
        combined = self.own(values)
        combined = combined + self.incoming(torch.sparse.mm(incoming, messages))
        combined = combined + self.outgoing(torch.sparse.mm(outgoing, messages))
        return F.relu(self.norm(combined))

class SnapshotClassifier(nn.Module):
    def __init__(self, input_width, hidden=48, dropout=0.15, kind="sage", residual=True, message_dropout=0.2):
        super().__init__()
        if kind not in {"sage", "mlp"}:
            raise ValueError("kind must be sage or mlp")
        self.kind = kind
        self.dropout = float(dropout)
        self.residual = bool(residual and kind == "sage")
        if kind == "sage":
            self.first = BidirectionalSageLayer(input_width, hidden, message_dropout)
            self.second = BidirectionalSageLayer(hidden, hidden, message_dropout)
        else:
            self.first = nn.Sequential(nn.Linear(input_width, hidden), nn.LayerNorm(hidden), nn.ReLU())
            self.second = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.output = nn.Linear(hidden, 1)
        if self.residual:
            self.direct = nn.Sequential(nn.Linear(input_width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, values, incoming=None, outgoing=None):
        if self.kind == "sage":
            hidden = self.first(values, incoming, outgoing)
            hidden = F.dropout(hidden, self.dropout, self.training)
            hidden = self.second(hidden, incoming, outgoing)
        else:
            hidden = self.first(values)
            hidden = F.dropout(hidden, self.dropout, self.training)
            hidden = self.second(hidden)
        hidden = F.dropout(hidden, self.dropout, self.training)
        logits = self.output(hidden).squeeze(1)
        if self.residual:
            logits = logits + self.direct(values).squeeze(1)
        return logits

def _snapshot_mean_tensor(adjacency, indices):
    matrix = adjacency[indices][:, indices].tocsr().astype(np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    matrix = sp.diags(inverse, format="csr") @ matrix
    coo = matrix.tocoo()
    coordinates = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(coordinates, values, size=matrix.shape).coalesce()

def _fit_feature_statistics(graph, fit_cutoff, clip):
    values = np.asarray(graph["x"], dtype=np.float32)
    times = np.asarray(graph["node_times"])
    supplied = np.asarray(graph["supplied_features"], dtype=bool)
    fitting = supplied & (times <= fit_cutoff)
    if not fitting.any():
        raise ValueError("No supplied feature rows in the fitting period")
    fitting_values = values[fitting].copy()
    fitting_values[~np.isfinite(fitting_values)] = np.nan
    if np.isnan(fitting_values).all(axis=0).any():
        raise ValueError("An entire feature is missing in the fitting period")
    median = np.nanmedian(fitting_values, axis=0)
    fitting_values = np.where(np.isnan(fitting_values), median, fitting_values)
    mean = fitting_values.mean(axis=0, dtype=np.float64)
    std = fitting_values.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return {"median": median, "mean": mean, "std": std, "clip": float(clip),
            "fit_feature_rows": int(fitting.sum())}

def _transform_features(values, supplied, statistics):
    values = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(values)
    missing = (~np.asarray(supplied, dtype=bool)) | invalid.any(axis=1)
    imputed = np.where(invalid, statistics["median"], values)
    standardized = (imputed - statistics["mean"]) / statistics["std"]
    standardized = np.clip(standardized, -statistics["clip"], statistics["clip"])
    # A missing-feature node has no numerical attributes; zero is a neutral
    # standardized placeholder, and the explicit final channel distinguishes it.
    standardized[~np.asarray(supplied, dtype=bool)] = 0.0
    return np.column_stack([standardized, missing.astype(np.float32)]).astype(np.float32)

def _validate_graph_model_inputs(graph, labels, fit_cutoff, prediction_indices, inner_validation_indices):
    values = np.asarray(graph["x"])
    times = np.asarray(graph["node_times"])
    supplied = np.asarray(graph["supplied_features"])
    labels = np.asarray(labels, dtype=float)
    n = len(values)
    if values.ndim != 2 or times.shape != (n,) or supplied.shape != (n,) or labels.shape != (n,):
        raise ValueError("Graph arrays do not align")
    if not np.isfinite(times).all() or not (times > 0).all():
        raise ValueError("Only anchored positive-time nodes may enter the model")
    for name in ["incoming", "outgoing"]:
        adjacency = graph[name]
        if not sp.isspmatrix_csr(adjacency) or adjacency.shape != (n, n):
            raise ValueError(f"{name} must be an aligned scipy CSR matrix")
        coo = adjacency.tocoo(copy=False)
        if not np.all(times[coo.row] == times[coo.col]):
            raise ValueError("Cross-time message passing is prohibited")
        if not np.isfinite(coo.data).all() or (coo.data < 0).any():
            raise ValueError("Adjacency must have finite nonnegative weights")
    finite_labels = np.isfinite(labels)
    if not set(np.unique(labels[finite_labels])).issubset({0.0, 1.0}):
        raise ValueError("Known labels must be zero or one")
    fitting = finite_labels & supplied & (times <= fit_cutoff)
    if len(np.unique(labels[fitting])) != 2:
        raise ValueError("Both supervised classes must occur in fitting rows")
    if np.any(finite_labels & ~supplied):
        raise ValueError("Feature-missing edge-only nodes cannot have supplied labels")
    for name, indices in [("prediction", prediction_indices), ("inner validation", inner_validation_indices)]:
        if indices is None:
            continue
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= n).any():
            raise ValueError(f"Invalid or duplicate {name} indices")
        if not np.all(times[indices] > fit_cutoff):
            raise ValueError(f"{name} must be strictly later than the fitting cutoff")
    if inner_validation_indices is not None:
        indices = np.asarray(inner_validation_indices, dtype=int)
        if not finite_labels[indices].all() or len(np.unique(labels[indices])) != 2:
            raise ValueError("Early stopping requires supplied labels from both classes")
    return labels, fitting

def fit_predict_graph_model(graph, labels, fit_cutoff, prediction_indices, config,
                            seed=42, epochs=None, inner_validation_indices=None):
    """Train on labelled fitting snapshots and predict later snapshots.

    With epochs=None, inner_validation_indices is mandatory and controls early
    stopping. With a fixed epoch count, no validation labels affect fitting.
    config keys: kind, hidden, dropout, residual, learning_rate, weight_decay,
    class_power, max_epochs, patience, min_epochs, snapshot_weighting, clip, message_dropout,
    threads, verbose. No external fit/prediction cache is read or written.
    """
    started = time.perf_counter()
    defaults = {"kind": "sage", "hidden": 48, "dropout": 0.15, "residual": True,
                "learning_rate": 0.003, "weight_decay": 0.0001, "class_power": 0.5,
                "max_epochs": 60, "patience": 10, "min_epochs": 8,
                "snapshot_weighting": "rows", "clip": 8.0, "threads": 2, "verbose": False,
                "message_dropout": 0.2}
    if set(config) - set(defaults):
        raise ValueError(f"Unrecognized graph model settings: {sorted(set(config) - set(defaults))}")
    cfg = {**defaults, **config}
    if cfg["snapshot_weighting"] not in {"rows", "equal_time"}:
        raise ValueError("snapshot_weighting must be rows or equal_time")
    if cfg["kind"] not in {"sage", "mlp"}:
        raise ValueError("kind must be sage or mlp")
    if not 0 <= cfg["message_dropout"] < 1 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Dropout rates must lie in [0,1)")
    if epochs is None and inner_validation_indices is None:
        raise ValueError("Inner validation is required when selecting the epoch count")
    if epochs is not None and inner_validation_indices is not None:
        raise ValueError("Fixed-epoch fitting cannot receive inner validation indices")
    total_epochs = int(cfg["max_epochs"] if epochs is None else epochs)
    if total_epochs < 1:
        raise ValueError("At least one training epoch is required")
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    if inner_validation_indices is not None:
        inner_validation_indices = np.asarray(inner_validation_indices, dtype=np.int64)
    labels, fitting = _validate_graph_model_inputs(graph, labels, fit_cutoff, prediction_indices, inner_validation_indices)
    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    statistics = _fit_feature_statistics(graph, fit_cutoff, cfg["clip"])
    node_times = np.asarray(graph["node_times"])
    supplied = np.asarray(graph["supplied_features"], dtype=bool)
    fit_steps = np.unique(node_times[fitting])
    needed_steps = set(fit_steps.tolist()) | set(node_times[prediction_indices].tolist())
    if inner_validation_indices is not None:
        needed_steps |= set(node_times[inner_validation_indices].tolist())
    snapshots = {}
    fit_counts = {}
    for step in sorted(needed_steps):
        indices = np.flatnonzero(node_times == step)
        # The MLP has no message passing; omitting feature-missing nodes is exact.
        if cfg["kind"] == "mlp":
            indices = indices[supplied[indices]]
        transformed = _transform_features(np.asarray(graph["x"])[indices], supplied[indices], statistics)
        local_fit = np.flatnonzero(fitting[indices])
        item = {"indices": indices, "x": torch.from_numpy(transformed), "supervised": torch.from_numpy(local_fit)}
        if len(local_fit):
            item["y"] = torch.from_numpy(labels[indices[local_fit]].astype(np.float32))
            fit_counts[int(step)] = len(local_fit)
        if cfg["kind"] == "sage":
            item["incoming"] = _snapshot_mean_tensor(graph["incoming"], indices)
            item["outgoing"] = _snapshot_mean_tensor(graph["outgoing"], indices)
        snapshots[int(step)] = item
    fit_indices = np.flatnonzero(fitting)
    class_weights = np.ones(len(fit_indices), dtype=float)
    if cfg["snapshot_weighting"] == "equal_time":
        class_weights = np.array([1.0 / fit_counts[int(step)] for step in node_times[fit_indices]])
    positive = class_weights[labels[fit_indices] == 1].sum()
    negative = class_weights[labels[fit_indices] == 0].sum()
    positive_multiplier = float((negative / positive) ** float(cfg["class_power"]))
    positive_tensor = torch.tensor(positive_multiplier, dtype=torch.float32)
    model = SnapshotClassifier(np.asarray(graph["x"]).shape[1] + 1, int(cfg["hidden"]),
                               cfg["dropout"], cfg["kind"], cfg["residual"], cfg["message_dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    mean_count = float(np.mean(list(fit_counts.values())))
    history = []
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0

    def forward_snapshot(item):
        return model(item["x"], item.get("incoming"), item.get("outgoing"))

    def predict_indices(indices):
        probability = np.empty(len(indices), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for step in np.unique(node_times[indices]):
                positions = np.flatnonzero(node_times[indices] == step)
                requested = indices[positions]
                item = snapshots[int(step)]
                local = np.searchsorted(item["indices"], requested)
                if (local >= len(item["indices"])).any() or not np.array_equal(item["indices"][local], requested):
                    raise ValueError("Prediction row is absent from its snapshot")
                logits = forward_snapshot(item)
                probability[positions] = torch.sigmoid(logits[torch.from_numpy(local)]).numpy()
        return probability

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        for step in rng.permutation(fit_steps):
            item = snapshots[int(step)]
            optimizer.zero_grad(set_to_none=True)
            logits = forward_snapshot(item)[item["supervised"]]
            loss = F.binary_cross_entropy_with_logits(logits, item["y"], pos_weight=positive_tensor)
            # Uniformly shuffled snapshot updates estimate the chosen objective.
            # Scaling by row count yields a row-weighted objective in expectation.
            if cfg["snapshot_weighting"] == "rows":
                loss = loss * (fit_counts[int(step)] / mean_count)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        record = {"epoch": epoch, "mean_training_loss": total_loss / len(fit_steps)}
        if inner_validation_indices is not None:
            validation_probability = predict_indices(inner_validation_indices)
            auc = float(roc_auc_score(labels[inner_validation_indices], validation_probability))
            record["inner_auc"] = auc
            if auc > best_auc + 1e-6:
                best_auc, best_epoch = auc, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            history.append(record)
            if cfg["verbose"]:
                print(f"{cfg['kind']} epoch {epoch}: inner AUC {auc:.6f}", flush=True)
            if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
                break
        else:
            history.append(record)
    if inner_validation_indices is not None:
        model.load_state_dict(best_state)
        used_epochs = best_epoch
    else:
        used_epochs = total_epochs
    probability = predict_indices(prediction_indices)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise AssertionError("Graph model returned invalid probabilities")
    info = {"kind": cfg["kind"], "seed": int(seed), "fit_cutoff": int(fit_cutoff),
            "fitting_labels": int(fitting.sum()), "fit_feature_rows": statistics["fit_feature_rows"],
            "fitting_snapshots": len(fit_steps), "positive_multiplier": positive_multiplier,
            "epochs_executed": len(history), "selected_epochs": used_epochs,
            "inner_best_auc": best_auc if inner_validation_indices is not None else None,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seconds": time.perf_counter() - started, "config": cfg, "history": history,
            "device": "CPU", "transformation": "fitting-only imputation/standardization plus missing-row flag"}
    np.savez_compressed(CACHE/f"{MODEL_TAG}_weights.npz",**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    joblib.dump(statistics,CACHE/f"{MODEL_TAG}_statistics.joblib")
    return probability, used_epochs, info

class TemporalFeatureTransform:
    """Fit raw, drop90, stable, or nuisance3 without target or future inputs.

    Pass only the fitting partition's labelled feature rows to fit. Targets are
    deliberately absent from this interface. transform uses frozen state and
    never learns from a prediction batch, its time steps, or future observations.

    drop90 removes features whose absolute fitting-row Spearman correlation with
    time is >=0.90. stable removes features at >=0.70 or whose between-snapshot
    median IQR exceeds ten times their typical within-snapshot IQR. When the
    denominator is zero, a positive numerator gives infinity; zero/zero is zero.
    Thus constant sparse features are not discarded merely for having zero IQR.

    nuisance3 estimates up to three major changing directions from equally
    weighted fitting-snapshot median vectors after fitting-only standardization.
    It subtracts each row's projection onto those directions. This removes linear
    nuisance directions and can also remove useful signal, so it is a candidate
    representation to validate, not a guaranteed invariant fraud representation.
    All 165 anonymous coordinates are considered; no feature meaning or ordering
    partition is assumed. The direction count and thresholds are prespecified.
    """

    def __init__(self, policy="raw"):
        if policy not in {"raw", "drop90", "stable", "nuisance3"}:
            raise ValueError("Unknown temporal feature policy.")
        self.policy = policy
        self.fitted_ = False

    @staticmethod
    def _values(X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("Features must form a nonempty two-dimensional matrix.")
        values = values.copy()
        values[~np.isfinite(values)] = np.nan
        return values

    def fit(self, X, times):
        values = self._values(X)
        times = np.asarray(times)
        if times.shape != (len(values),) or not np.isfinite(times).all():
            raise ValueError("Fitting times must be finite and align with fitting rows.")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("Cannot estimate a feature absent from every fitting row.")
        self.n_features_in_ = values.shape[1]
        self.kept_ = np.arange(self.n_features_in_)
        self.removed_ = np.empty(0, dtype=np.int64)
        self.time_spearman_ = np.zeros(self.n_features_in_)
        self.between_within_ratio_ = np.zeros(self.n_features_in_)
        self.components_ = np.empty((0, self.n_features_in_))
        frame = pd.DataFrame(values)
        if self.policy in {"drop90", "stable"}:
            # Pearson correlation of average ranks equals Spearman correlation.
            self.time_spearman_ = frame.rank().corrwith(pd.Series(times).rank()).fillna(0).to_numpy()
            threshold = .90 if self.policy == "drop90" else .70
            remove = np.abs(self.time_spearman_) >= threshold
            if self.policy == "stable":
                grouped = frame.groupby(times)
                medians = grouped.median()
                between = (medians.quantile(.75) - medians.quantile(.25)).to_numpy()
                within = (grouped.quantile(.75) - grouped.quantile(.25)).median(axis=0).to_numpy()
                ratio = np.divide(between, within, out=np.zeros_like(between), where=within > 0)
                ratio[(within == 0) & (between > 0)] = np.inf
                self.between_within_ratio_ = ratio
                remove |= ratio > 10.0
            self.removed_ = np.flatnonzero(remove)
            self.kept_ = np.flatnonzero(~remove)
            if len(self.kept_) == 0:
                raise ValueError("The fixed temporal stability rule removed every feature.")
        elif self.policy == "nuisance3":
            if len(np.unique(times)) < 2:
                raise ValueError("Temporal nuisance estimation requires at least two fitting snapshots.")
            self.imputation_medians_ = np.nanmedian(values, axis=0)
            filled = np.where(np.isfinite(values), values, self.imputation_medians_)
            self.mean_ = filled.mean(axis=0)
            self.scale_ = filled.std(axis=0, ddof=0)
            self.scale_[self.scale_ <= np.finfo(np.float64).eps] = 1.0
            standardized = (filled - self.mean_) / self.scale_
            snapshot_medians = pd.DataFrame(standardized).groupby(times).median().to_numpy()
            self.direction_center_ = snapshot_medians.mean(axis=0)
            centered = snapshot_medians - self.direction_center_
            _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
            tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values.max(initial=0)
            directions = min(3, int((singular_values > tolerance).sum()))
            self.components_ = right[:directions]
            total = (singular_values ** 2).sum()
            self.snapshot_variance_removed_ = float((singular_values[:directions] ** 2).sum() / total) if total > 0 else 0.0
        self.info_ = {
            "policy": self.policy,
            "fit_rows": len(values),
            "fit_min_time": int(np.min(times)),
            "fit_max_time": int(np.max(times)),
            "fit_snapshots": int(len(np.unique(times))),
            "input_features": self.n_features_in_,
            "output_features": len(self.kept_),
            "removed_features": [f"feat_{j + 1}" for j in self.removed_],
            "time_spearman_threshold": .90 if self.policy == "drop90" else (.70 if self.policy == "stable" else None),
            "between_within_ratio_threshold": 10.0 if self.policy == "stable" else None,
            "nuisance_directions": len(self.components_),
            "snapshot_variance_removed": getattr(self, "snapshot_variance_removed_", None),
            "prediction_batch_statistics_used": False,
        }
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Fit the temporal feature transform first.")
        values = self._values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from the fitting width.")
        if self.policy == "nuisance3":
            filled = np.where(np.isfinite(values), values, self.imputation_medians_)
            standardized = (filled - self.mean_) / self.scale_
            centered = standardized - self.direction_center_
            result = standardized - (centered @ self.components_.T) @ self.components_
        else:
            result = values[:, self.kept_]
        return result.astype(np.float32)

    def fit_transform(self, X, times):
        return self.fit(X, times).transform(X)

class FitOnlyNumericalTransform:
    """Fit imputation and scaling on the fitting matrix, then transform copies."""

    def __init__(self, method="quantile", seed=42, n_quantiles=256, clip=8.0):
        if method not in {"standard", "quantile"}:
            raise ValueError("Preprocessing must be standard or quantile.")
        if int(n_quantiles) < 2 or not np.isfinite(clip) or clip <= 0:
            raise ValueError("Quantile count and clipping threshold must be positive.")
        self.method = method
        self.seed = int(seed)
        self.n_quantiles = int(n_quantiles)
        self.clip = float(clip)

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("Fitting features must be a nonempty numerical matrix.")
        clean = values.copy()
        clean[~np.isfinite(clean)] = np.nan
        if np.isnan(clean).all(axis=0).any():
            raise ValueError("An entire fitting feature is missing; inspect it before fitting.")
        self.median_ = np.nanmedian(clean, axis=0)
        clean = np.where(np.isnan(clean), self.median_, clean)
        self.n_features_in_ = clean.shape[1]
        self.fit_rows_ = len(clean)
        if self.method == "standard":
            self.scaler_ = StandardScaler()
        else:
            self.scaler_ = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(clean)),
                output_distribution="normal",
                subsample=max(100_000, len(clean)),
                random_state=self.seed,
                copy=True,
            )
        self.scaler_.fit(clean)
        state = hashlib.sha256(self.median_.tobytes())
        for name in ["mean_", "scale_", "quantiles_", "references_"]:
            if hasattr(self.scaler_, name):
                state.update(np.asarray(getattr(self.scaler_, name)).tobytes())
        self.state_sha256_ = state.hexdigest()
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or len(values) == 0:
            raise ValueError("Prediction features must align with fitting columns.")
        clean = np.where(np.isfinite(values), values, self.median_)
        result = self.scaler_.transform(clean)
        result = np.clip(result, -self.clip, self.clip).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Numerical preprocessing returned a nonfinite value.")
        return np.ascontiguousarray(result)

class SharedEnsembleLinear(nn.Module):
    """A shared dense weight matrix with member-specific input/output scales."""

    def __init__(self, input_width, output_width, members):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_width, input_width))
        self.input_scale = nn.Parameter(torch.empty(members, input_width))
        self.output_scale = nn.Parameter(torch.empty(members, output_width))
        self.bias = nn.Parameter(torch.zeros(members, output_width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Independent signs break the symmetry between ensemble members.
        with torch.no_grad():
            self.input_scale.copy_(torch.randint(0, 2, self.input_scale.shape) * 2 - 1)
            self.output_scale.copy_(torch.randint(0, 2, self.output_scale.shape) * 2 - 1)

    def forward(self, values):
        return F.linear(values * self.input_scale, self.weight) * self.output_scale + self.bias

class ParameterEfficientMLP(nn.Module):
    """Two shared hidden layers and a separate classification head per member."""

    def __init__(self, n_features, k=8, width=64, dropout=0.05):
        super().__init__()
        self.k = int(k)
        self.first = SharedEnsembleLinear(n_features, width, self.k)
        self.second = SharedEnsembleLinear(width, width, self.k)
        self.dropout = float(dropout)
        self.output_weight = nn.Parameter(torch.empty(self.k, width))
        self.output_bias = nn.Parameter(torch.zeros(self.k))
        nn.init.normal_(self.output_weight, mean=0.0, std=1.0 / math.sqrt(width))

    def forward(self, values):
        hidden = values[:, None, :].expand(-1, self.k, -1)
        hidden = F.relu(self.first(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.second(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return (hidden * self.output_weight).sum(dim=-1) + self.output_bias

def _binary_targets(values, rows, name):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain one finite known target per row.")
    if set(np.unique(values)) != {0.0, 1.0}:
        raise ValueError(f"{name} must contain both classes, encoded as 0 and 1.")
    return values

def fit_predict_ensemble(X_fit, y_fit, X_pred, config=None, seed=42, epochs=None,
                         X_inner=None, y_inner=None):
    """Return (class-1 probability, selected epochs, aggregate fit metadata).

    epochs=None requires an explicitly supplied inner temporal validation set.
    Fixed-epoch fitting rejects either inner argument. Prediction labels are not
    an API input. Imputation/scaling fit only X_fit in both modes. The caller must
    supply temporal partitions and keep inner rows inside the outer fitting
    period. Each ensemble head receives its own binary loss; inference averages
    head probabilities, not logits. Unknown labels cannot enter supervised loss.
    """
    started = time.perf_counter()
    defaults = {
        "k": 8, "width": 64, "dropout": 0.05,
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "batch_size": 512, "preprocessing": "quantile", "class_power": 0.5,
        "max_epochs": 80, "patience": 12, "min_epochs": 8,
        "threads": 2, "n_quantiles": 256, "clip": 8.0, "verbose": False,
    }
    config = {} if config is None else dict(config)
    if set(config) - set(defaults):
        raise ValueError(f"Unknown ensemble settings: {sorted(set(config) - set(defaults))}")
    cfg = {**defaults, **config}
    for key in ["k", "width", "batch_size", "max_epochs", "patience", "min_epochs", "threads"]:
        if int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if not 0 <= cfg["dropout"] < 1 or not 0 <= cfg["class_power"] <= 1:
        raise ValueError("Dropout and class weighting power are outside their allowed ranges.")
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Optimizer settings must be positive/nonnegative.")
    early = epochs is None
    if early and (X_inner is None or y_inner is None):
        raise ValueError("An inner temporal validation set is required to select epochs.")
    if not early and (X_inner is not None or y_inner is not None):
        raise ValueError("Fixed-epoch fits cannot receive inner validation features or labels.")
    total_epochs = cfg["max_epochs"] if early else epochs
    if int(total_epochs) != total_epochs or total_epochs < 1:
        raise ValueError("Epoch count must be a positive integer.")
    total_epochs = int(total_epochs)
    X_fit = np.asarray(X_fit)
    if X_fit.ndim != 2:
        raise ValueError("Fitting features must be a numerical matrix.")
    y = _binary_targets(y_fit, len(X_fit), "Fitting targets")
    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(int(seed))
    batch_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 1009)
    transform = FitOnlyNumericalTransform(
        cfg["preprocessing"], seed, cfg["n_quantiles"], cfg["clip"]
    ).fit(X_fit)
    fit_tensor = torch.from_numpy(transform.transform(X_fit))
    pred_tensor = torch.from_numpy(transform.transform(X_pred))
    target_tensor = torch.from_numpy(y)
    if early:
        inner_tensor = torch.from_numpy(transform.transform(X_inner))
        inner_targets = _binary_targets(y_inner, len(inner_tensor), "Inner targets")
    else:
        inner_tensor = inner_targets = None
    ratio = float(((y == 0).sum() / (y == 1).sum()) ** cfg["class_power"])
    positive_weight = torch.tensor(ratio, dtype=torch.float32)
    model = ParameterEfficientMLP(
        X_fit.shape[1], int(cfg["k"]), int(cfg["width"]), cfg["dropout"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    history = []
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    observed_rss = psutil.Process().memory_info().rss / 2**30

    def probability(values):
        model.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(values), int(cfg["batch_size"])):
                logits = model(values[start:start + int(cfg["batch_size"])])
                result.append(torch.sigmoid(logits).mean(dim=1).numpy())
        scores = np.concatenate(result).astype(np.float64)
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise RuntimeError("The neural ensemble returned invalid probabilities.")
        return scores

    for epoch in range(1, total_epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=batch_rng)
        total_loss = 0.0
        for start in range(0, len(y), int(cfg["batch_size"])):
            indices = order[start:start + int(cfg["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit_tensor[indices])
            targets = target_tensor[indices, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite neural training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        record = {"epoch": epoch, "training_loss": total_loss / len(y)}
        observed_rss = max(observed_rss, psutil.Process().memory_info().rss / 2**30)
        if early:
            auc = float(roc_auc_score(inner_targets, probability(inner_tensor)))
            record["inner_auc"] = auc
            if auc > best_auc + 1e-6:
                best_auc, best_epoch = auc, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(record)
        if cfg["verbose"]:
            print(f"Custom ensemble epoch {epoch}: {record}", flush=True)
        if early and epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
            break
    if early:
        if best_state is None:
            raise RuntimeError("Inner validation did not produce a finite best checkpoint.")
        model.load_state_dict(best_state)
        used = best_epoch
    else:
        used = total_epochs
    prediction = probability(pred_tensor)
    info = {
        "architecture": "custom two-layer BatchEnsemble-style MLP; not exact TabM",
        "seed": int(seed), "fit_rows": len(y), "prediction_rows": len(prediction),
        "n_features": X_fit.shape[1], "parameters": sum(p.numel() for p in model.parameters()),
        "positive_multiplier": ratio, "preprocessing": cfg["preprocessing"],
        "transform_fit_rows": transform.fit_rows_, "transform_sha256": transform.state_sha256_,
        "epochs_executed": len(history), "selected_epochs": int(used),
        "inner_best_auc": float(best_auc) if early else None,
        "seconds": time.perf_counter() - started, "rss_gib": observed_rss,
        "device": "CPU", "config": cfg, "history": history,
    }
    np.savez_compressed(CACHE/f"{MODEL_TAG}_weights.npz",**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    joblib.dump(transform,CACHE/f"{MODEL_TAG}_transform.joblib")
    return prediction, int(used), info

def log(info,tag):
    record={k:v for k,v in info.items() if k not in ['history']};record['tag']=tag;logs.append(record);save('fit_log.json',logs)
    print(tag,'seconds',round(info['seconds'],2),flush=True)
