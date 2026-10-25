"""Rebuild the fixed V22 submission from the four original competition files.

No earlier predictions, model checkpoints, experiment directories or external
labels are inputs. Generated checkpoints belong only to this reproduction run.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import reference_components as rc
from . import target_graph as tg

SEEDS = [42, 52, 62]
FEATURES = [f"feat_{i}" for i in range(1, 166)]
RAW_HASHES = {
    "train.csv": "6f7e56883f0aa27147262a0d71954ad64702ca57f16fbf9ed58fae21a66dd706",
    "test.csv": "344e47c8cd8815f4072d7f6ecbecee8919ab648294a8eb4ed032a026c3eb91db",
    "txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
    "sample_submission.csv": "3dba74ad6151053aa21f366fb9faf6da7393969a0ba6f8a2ebcec561871f739e",
}
EDGE_PARAMS = dict(n_estimators=350, max_depth=3, learning_rate=.04,
    min_child_weight=15, reg_lambda=20, reg_alpha=.5, subsample=.85,
    colsample_bytree=.85, max_bin=128, tree_method="hist", base_score=.5, n_jobs=2)
RECENT_PARAMS = dict(n_estimators=800, max_depth=5, learning_rate=.03,
    min_child_weight=5, reg_lambda=10, reg_alpha=.1, subsample=.9,
    colsample_bytree=.85, max_bin=128, tree_method="hist", base_score=.5,
    n_jobs=2, eval_metric="logloss")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def rank(values):
    return (rankdata(values, method="average") - .5) / len(values)


def load_data(data_dir):
    """Reconstruct same-time topology using features and edges, never labels."""
    raw = Path(data_dir)
    for name, expected in RAW_HASHES.items():
        if not (raw / name).is_file():
            raise FileNotFoundError(f"Place the original competition file at {raw/name}")
        if sha(raw / name) != expected:
            raise ValueError(f"Input hash differs from the competition version: {name}")
    train, test = pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "test.csv")
    template = pd.read_csv(raw / "sample_submission.csv")
    edges = pd.read_csv(raw / "txs_edgelist.csv")
    np.testing.assert_array_equal(template["index"], test["index"])
    assert test["index"].is_unique and len(test) == 15329
    nodes = pd.concat([train[["txId", "time_step"] + FEATURES], test[["txId", "time_step"] + FEATURES]], ignore_index=True)
    supplied_ids = pd.Index(nodes.txId)
    endpoints = pd.Index(pd.unique(edges[["txId1", "txId2"]].to_numpy().ravel()))
    all_ids = supplied_ids.append(endpoints.difference(supplied_ids).sort_values())
    source, destination = all_ids.get_indexer(edges.txId1), all_ids.get_indexer(edges.txId2)
    incoming = sp.csr_matrix((np.ones(len(edges), np.float32), (destination, source)), shape=(len(all_ids), len(all_ids)))
    assert incoming.nnz == len(edges) and (source != destination).all()
    count, component = connected_components(incoming, directed=False)
    low, high = np.full(count, np.inf), np.full(count, -np.inf)
    np.minimum.at(low, component[:len(nodes)], nodes.time_step)
    np.maximum.at(high, component[:len(nodes)], nodes.time_step)
    # For these exact files every component has one observed time anchor.
    assert np.isfinite(low).all() and np.array_equal(low, high)
    times = low[component].astype(np.int32)
    np.testing.assert_array_equal(times[:len(nodes)], nodes.time_step)
    assert np.array_equal(times[source], times[destination])
    x64 = nodes[FEATURES].to_numpy(np.float64)
    graph_x = np.full((len(all_ids), 165), np.nan, np.float32)
    graph_x[:len(nodes)] = x64
    supplied = np.arange(len(all_ids)) < len(nodes)
    labels = np.full(len(all_ids), np.nan)
    labels[:len(train)] = train.label.to_numpy()
    known = np.isfinite(labels)
    assert np.isin(labels[known], [0, 1]).all() and known.sum() == 31235
    target_visible = known.copy()
    target_visible[len(train):len(nodes)] = True
    graph = dict(x=graph_x, node_times=times, supplied_features=supplied,
                 incoming=incoming, outgoing=incoming.T.tocsr())
    return dict(train=train, test=test, template=template, x64=x64, graph=graph,
                labels=labels, times=times, known=known, target_visible=target_visible,
                node_ids=all_ids.to_numpy(), source=source, destination=destination)


def sibling_graph(incoming, ids):
    """Connect children of a shared parent with 2..32 visible children."""
    visible = incoming[ids].tocsc()
    degree = np.asarray(visible.sum(axis=0)).ravel()
    keep = (degree >= 2) & (degree <= 32)
    visible = visible[:, keep]
    relation = (visible @ sp.diags(1 / (degree[keep]-1)) @ visible.T).tocsr()
    relation.setdiag(0)
    relation.eliminate_zeros()
    return relation


def pair_features(ids, source, destination, values, degrees):
    a, b = values[ids[source]], values[ids[destination]]
    da, db = degrees[ids[source]], degrees[ids[destination]]
    return np.column_stack([np.abs(a-b), np.abs(da-db), np.minimum(a,b),
        np.maximum(a,b), np.minimum(da,db), np.maximum(da,db)]).astype(np.float32)


def train_predict(context, cutoff, query, output_dir):
    """Fit the frozen final recipe at one historical or final cutoff.

    Historical evaluation is retrospective: recipe choices have already used
    these time periods. It is not a fresh holdout or the original adaptive CV.
    """
    started = time.perf_counter()
    cache = Path(output_dir) / f"c{cutoff}"
    cache.mkdir(parents=True, exist_ok=True)
    query = np.asarray(query, dtype=int)
    y, times, graph, x64 = (context[k] for k in ["labels", "times", "graph", "x64"])
    fit = np.flatnonzero(context["known"] & (times <= cutoff))
    assert len(query) and np.all(times[query] > cutoff) and len(fit)
    identity = dict(cutoff=cutoff, query_sha256=hashlib.sha256(query.tobytes()).hexdigest(), raw_hashes=RAW_HASHES, seeds=SEEDS)
    if (cache / "inputs.json").exists():
        assert json.loads((cache / "inputs.json").read_text()) == identity
    else:
        save(cache / "inputs.json", identity)
    rc.CACHE = cache
    logs = []

    def component(tag, function):
        rc.MODEL_TAG = tag
        path = cache / f"{tag}_prediction.npy"
        if path.exists():
            return np.load(path)
        prediction, _, info = function()
        np.save(path, prediction)
        save(cache / f"{tag}_fit.json", {k:v for k,v in info.items() if k != "history"})
        logs.append(tag)
        print(f"Completed {tag}", flush=True)
        return prediction

    runner = rc.TreeRunner(x64, np.empty((len(x64),0)), x64, x64, y[:len(x64)], times[:len(x64)], cache)
    with threadpool_limits(limits=2):
        e04 = component("E04", lambda: runner.fit(dict(id="E04", reference="E04"), cutoff, query, rounds=400, stage="fixed_reproduction"))
        cats = []
        for policy, rounds in [("raw",336), ("drop",244)]:
            for seed in SEEDS:
                cats.append(component(f"CAT_{policy}_s{seed}", lambda policy=policy, rounds=rounds, seed=seed: runner.fit(dict(id=f"CAT_{policy}",reference="CAT",policy=policy),cutoff,query,seed=seed,rounds=rounds,stage="fixed_reproduction")))
        v4 = .5*e04 + .5*np.mean(cats,axis=0)
        sage = []
        for seed in SEEDS:
            sage.append(component(f"SAGE_s{seed}", lambda seed=seed: rc.fit_predict_graph_model(graph,np.where(times<=cutoff,y,np.nan),cutoff,query,
                dict(kind="sage",hidden=64,dropout=.15,message_dropout=.2,learning_rate=.003,weight_decay=.0001,class_power=.5,max_epochs=120,patience=20,min_epochs=8,threads=2),seed=seed,epochs=40)))
        v5 = .75*v4 + .25*np.mean(sage,axis=0)
        policy = rc.TemporalFeatureTransform("drop90").fit(x64[fit],times[fit])
        joblib.dump(policy,cache/"C37_feature_policy.joblib")
        a, b = policy.transform(x64[fit]), policy.transform(x64[query])
        neural = []
        for seed in SEEDS:
            neural.append(component(f"C37_s{seed}", lambda seed=seed: rc.fit_predict_ensemble(a,y[fit],b,
                dict(k=8,width=128,preprocessing="quantile",class_power=0.,learning_rate=.001,weight_decay=.0001,batch_size=512,max_epochs=80,patience=12,min_epochs=8,threads=2,dropout=.05),seed=seed,epochs=24)))
        c37 = np.mean(neural,axis=0)
        v9 = .75*rank(v5) + .25*rank(c37)
        tg.CACHE=cache; tg.save=lambda name,value:save(cache/name,value); tg.sha=sha
        tg.X=graph["x"]; tg.node_times=times; tg.graph_labels=y
        tg.all_visible=graph["supplied_features"]; tg.target_visible=context["target_visible"]
        tg.query=lambda lo,hi:np.flatnonzero(context["known"]&(times>=lo)&(times<=hi))
        tg.PROTOCOL=dict(profiles={"TARGET_QNT":dict(availability="targets",filter="drop90",transform="quantile")})
        tg.transforms={};tg.predictions={};tg.fits=[]
        tg.snapshots={int(t):np.flatnonzero(times==t) for t in np.unique(times)}
        tg.adjacency={t:(tg._snapshot_mean_tensor(graph["incoming"],rows),tg._snapshot_mean_tensor(graph["outgoing"],rows)) for t,rows in tg.snapshots.items()}
        target_parts=[]
        for seed in SEEDS:
            path=cache/f"TARGET_QNT_c{cutoff}_s{seed}_predictions.npz"
            if path.exists():
                stored=np.load(path); np.testing.assert_array_equal(stored["indices"],query); part=stored["prediction"]
            else:
                part=tg.fit_one("TARGET_QNT",cutoff,seed,query,"fixed_reproduction")
                logs.append(f"TARGET_QNT_s{seed}")
            target_parts.append(part)
        v14=.9*v9+.1*rank(np.mean(target_parts,axis=0))
        x=x64.astype(np.float32)
        corr=pd.DataFrame(x[fit]).corrwith(pd.Series(times[fit]),method="spearman").fillna(0).to_numpy()
        columns=np.flatnonzero(np.abs(corr)<.90)
        weights=2.**(-(cutoff-times[fit])/10);weights/=weights.mean()
        recent=[]; selector=None
        for seed in SEEDS:
            model=XGBClassifier(**RECENT_PARAMS,random_state=seed)
            path=cache/f"RECENT_s{seed}.ubj"
            if path.exists():model.load_model(path)
            else:
                model.fit(x[np.ix_(fit,columns)],y[fit].astype(int),sample_weight=weights);model.save_model(path)
                logs.append(f"RECENT_s{seed}")
            recent.append(model.predict(x[np.ix_(query,columns)],output_margin=True))
            if seed==42:selector=model
        incoming=graph["incoming"].astype(np.float64)
        direct=incoming[query][:,query].tocsr()
        degree=np.asarray(direct.sum(axis=1)).ravel()
        anchor=.95*v14+.05*rank(np.mean(recent,axis=0))
        margins=logit(np.clip(anchor,1e-8,1-1e-8))
        parent=np.divide(direct@margins,degree,out=margins.copy(),where=degree>0)
        v21=expit(.9*margins+.1*parent)
        selected=columns[np.argsort(selector.feature_importances_)[-32:]]
        qt=QuantileTransformer(n_quantiles=min(1000,len(fit)),output_distribution="normal",subsample=None,random_state=42).fit(x[np.ix_(fit,selected)])
        transformed=np.clip(qt.transform(x[:,selected]),-4,4)
        degrees=np.log1p(np.column_stack([np.asarray(incoming.sum(axis=1)).ravel(),np.asarray(incoming.sum(axis=0)).ravel()]))
        joblib.dump(dict(quantiles=qt,columns=selected,fit_indices=fit),cache/"edge_transform.joblib")
        np.savez_compressed(cache/"recent_state.npz",indices=fit,columns=columns,weights=weights)
        predictions={}
        for kind in ["direct","siblings"]:
            fit_graph=incoming[fit][:,fit] if kind=="direct" else sp.triu(sibling_graph(incoming,fit),k=1)
            f=fit_graph.tocoo();src,dst=f.col,f.row
            labels=y[fit].astype(int);same=(labels[src]==labels[dst])
            if kind=="direct":
                cells=2*labels[src]+labels[dst];weight=1/np.bincount(cells,minlength=4)[cells]
            else:
                cells=np.where(same,labels[src],2);totals=np.bincount(cells,weights=f.data,minlength=3)
                weight=f.data*np.array([.25,.25,.5])[cells]/totals[cells]
            weight/=weight.mean()
            fm=pair_features(fit,src,dst,transformed,degrees)
            relation=(direct if kind=="direct" else sibling_graph(incoming,query)).tocoo()
            qm=pair_features(query,relation.col,relation.row,transformed,degrees)
            parts=[]
            for seed in SEEDS:
                model=XGBClassifier(**EDGE_PARAMS,random_state=seed)
                path=cache/f"{kind}_s{seed}.ubj"
                if path.exists():model.load_model(path)
                else:
                    model.fit(fm,same.astype(int),sample_weight=weight);model.save_model(path)
                    logs.append(f"{kind}_s{seed}")
                parts.append(model.predict_proba(qm)[:,1])
            probability=np.mean(parts,axis=0)
            weighted=sp.csr_matrix((relation.data*np.maximum(2*probability-1,0),(relation.row,relation.col)),shape=relation.shape)
            if kind=="direct":
                weighted=weighted+weighted.T
                count=np.asarray(direct.sum(axis=1)).ravel()+np.asarray(direct.sum(axis=0)).ravel()
                own=logit(np.clip(v21,1e-8,1-1e-8));current=own.copy()
                mass=np.asarray(weighted.sum(axis=1)).ravel()
                for _ in range(3):current=own+.6*(weighted@current-mass*own)/np.maximum(count,1)
                risk=expit(current)
            else:
                count=np.asarray(relation.tocsr().sum(axis=1)).ravel()
                own=logit(np.clip(risk,1e-8,1-1e-8))
                risk=expit(own+.2*(weighted@own-np.asarray(weighted.sum(axis=1)).ravel()*own)/np.maximum(count,1))
            predictions[kind]=risk.copy()
            np.savez_compressed(cache/f"{kind}_edges.npz",indices=query,source=relation.col,destination=relation.row,multiplicity=relation.data,seed_probabilities=np.array(parts),probability=probability)
        assert np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all()
        np.savez_compressed(cache/"predictions.npz",indices=query,E04=e04,V4=v4,V5=v5,C37=c37,V9=v9,GRAPH=np.mean(target_parts,axis=0),V14=v14,V21=v21,V22=risk)
        summary=dict(cutoff=cutoff,known_fit_rows=len(fit),query_rows=len(query),seeds=SEEDS,seconds=time.perf_counter()-started,raw_hashes=RAW_HASHES,new_model_fits_this_invocation=len(logs),
            validation="Retrospective fixed final recipe on reused periods; not original adaptive metrics or an untouched holdout.")
        if np.isfinite(y[query]).all():summary["auc"]=float(roc_auc_score(y[query],risk))
        save(cache/"summary.json",summary)
        print(json.dumps(summary,indent=2),flush=True)
    gc.collect()
    return risk,summary


def generate_submission(context, output_dir):
    query=np.arange(len(context["train"]),len(context["train"])+len(context["test"]))
    risk,summary=train_predict(context,35,query,output_dir)
    frame=pd.DataFrame({"index":context["test"]["index"].to_numpy(),"target":risk})
    np.testing.assert_array_equal(frame["index"],context["template"]["index"])
    path=Path(output_dir)/"submission_edgegate_v22_reproduced.csv"
    frame.to_csv(path,index=False,float_format="%.17g")
    reloaded=pd.read_csv(path,float_precision="round_trip")
    assert list(reloaded.columns)==["index","target"] and reloaded["index"].is_unique
    np.testing.assert_array_equal(reloaded["index"],frame["index"])
    np.testing.assert_array_equal(reloaded.target,risk)
    save(Path(output_dir)/"submission_checks.json",dict(passed=True,rows=len(frame),columns=list(frame.columns),sha256=sha(path),minimum=float(risk.min()),maximum=float(risk.max()),template_order_exact=True,csv_roundtrip_exact=True))
    return path,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",default="data/raw")
    parser.add_argument("--output-dir",default="outputs/v22_reproduction")
    parser.add_argument("--stage",choices=["final","validate","all"],default="all")
    args=parser.parse_args();context=load_data(args.data_dir)
    if args.stage in ["validate","all"]:
        ids=np.flatnonzero(context["known"]&(context["times"]>30)&(context["times"]<=35))
        train_predict(context,30,ids,args.output_dir)
    if args.stage in ["final","all"]:generate_submission(context,args.output_dir)


if __name__=="__main__":main()
