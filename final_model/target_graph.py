# Frozen E18 TARGET_QNT training implementation; globals are set by the driver.
# The exploratory search's wall-time/fit-count cap is omitted: the final driver
# has exactly three predetermined seeds per cutoff, including on slower CPUs.
from pathlib import Path
import gc,hashlib,json,time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
import psutil,torch
from torch import nn
from torch.nn import functional as F
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

def fit_transform(profile,cutoff):
    key=f"{profile}_c{cutoff}"
    if key in transforms:return transforms[key]
    cfg=PROTOCOL["profiles"][profile]
    visible=all_visible if cfg["availability"]=="all_supplied" else target_visible
    fitrows=np.flatnonzero(visible&(node_times<=cutoff))
    values=X[fitrows].astype(np.float64)
    median=np.nanmedian(values,axis=0)
    assert np.isfinite(median).all()
    values=np.where(np.isfinite(values),values,median)
    keep=np.arange(165);corr=np.zeros(165)
    if cfg["filter"]=="drop90":
        ranked=rankdata(values,axis=0);rt=rankdata(node_times[fitrows])
        a=ranked-ranked.mean(axis=0);b=rt-rt.mean()
        denom=np.sqrt((a*a).sum(axis=0)*(b*b).sum())
        corr=np.divide((a*b[:,None]).sum(axis=0),denom,out=np.zeros(165),where=denom>0)
        keep=np.flatnonzero(np.abs(corr)<.90)
    selected=values[:,keep]
    state=dict(profile=profile,fit_cutoff=cutoff,fit_feature_rows=len(fitrows),
               fit_feature_indices_sha256=hashlib.sha256(fitrows.tobytes()).hexdigest(),
               median=median.tolist(),keep=keep.tolist(),time_correlations=corr.tolist(),
               method=cfg["transform"],clip=8)
    if cfg["transform"]=="standard":
        mean=selected.mean(axis=0);std=selected.std(axis=0);std[std<1e-8]=1
        state.update(mean=mean.tolist(),std=std.tolist())
    else:
        qt=QuantileTransformer(n_quantiles=min(256,len(selected)),output_distribution="normal",
                               subsample=max(100000,len(selected)),random_state=42).fit(selected)
        state.update(quantiles=qt.quantiles_.tolist(),references=qt.references_.tolist())
    save(f"{key}_transform.json",state)
    transforms[key]=state
    return state

def transform_nodes(state,rows,values_override=None):
    cfg=PROTOCOL["profiles"][state["profile"]]
    visible=(all_visible if cfg["availability"]=="all_supplied" else target_visible)[rows]
    values=X[rows] if values_override is None else values_override
    values=np.where(np.isfinite(values),values,np.asarray(state["median"]))[:,state["keep"]]
    if state["method"]=="standard":
        result=(values-np.asarray(state["mean"]))/np.asarray(state["std"])
    else:
        qt=QuantileTransformer(output_distribution="normal")
        qt.quantiles_=np.asarray(state["quantiles"]);qt.references_=np.asarray(state["references"])
        qt.n_quantiles_=len(qt.references_);qt.n_features_in_=len(state["keep"])
        result=qt.transform(values)
    result=np.clip(result,-8,8)
    result[~visible]=0
    result=np.column_stack([result,(~visible).astype(np.float32)]).astype(np.float32)
    assert np.isfinite(result).all()
    return result

def model_prediction(model,items,q):
    result=np.empty(len(q),np.float64);model.eval()
    with torch.no_grad():
        for step in np.unique(node_times[q]):
            pos=np.flatnonzero(node_times[q]==step);item=items[int(step)]
            local=np.searchsorted(item["rows"],q[pos])
            np.testing.assert_array_equal(item["rows"][local],q[pos])
            logits=model(item["x"],*adjacency[int(step)])
            result[pos]=torch.sigmoid(logits[torch.from_numpy(local)]).numpy()
    assert np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
    return result

def fit_one(profile,cutoff,seed,q,stage):
    key=f"{profile}_c{cutoff}_s{seed}"
    if key in predictions:
        original,p=predictions[key];pos=np.searchsorted(original,q)
        np.testing.assert_array_equal(original[pos],q)
        return p[pos]
    started=time.perf_counter()
    assert (node_times[q]>cutoff).all() and target_visible[q].all()
    fitting_labels=np.where(node_times<=cutoff,graph_labels,np.nan)
    fitting=np.isfinite(fitting_labels)
    assert not np.isfinite(fitting_labels[node_times>cutoff]).any()
    assert fitting.sum()==len(query(1,cutoff))
    state=fit_transform(profile,cutoff)
    needed=np.unique(np.r_[node_times[fitting],node_times[q]])
    items={}
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
        for step in needed:
            rows=snapshots[int(step)];local=np.flatnonzero(fitting[rows])
            items[int(step)]=dict(rows=rows,x=torch.from_numpy(transform_nodes(state,rows)),
                                  supervised=torch.from_numpy(local),
                                  y=torch.from_numpy(fitting_labels[rows[local]].astype(np.float32)))
        model=SnapshotClassifier(len(state["keep"])+1,hidden=64,dropout=.15,kind="sage",residual=True,message_dropout=.2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
        fitsteps=np.unique(node_times[fitting]);mean_count=fitting.sum()/len(fitsteps)
        ratio=float((np.sum(fitting_labels[fitting]==0)/np.sum(fitting_labels[fitting]==1))**.5)
        weight=torch.tensor(ratio,dtype=torch.float32)
        losses=[];rss=psutil.Process().memory_info().rss/2**30
        for epoch in range(40):
            model.train();loss_total=0.
            for step in rng.permutation(fitsteps):
                item=items[int(step)]
                optimizer.zero_grad(set_to_none=True)
                logits=model(item["x"],*adjacency[int(step)])[item["supervised"]]
                loss=F.binary_cross_entropy_with_logits(logits,item["y"],pos_weight=weight)
                loss=loss*(len(item["y"])/mean_count)
                assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();loss_total+=float(loss.detach())
            losses.append(loss_total/len(fitsteps))
            rss=max(rss,psutil.Process().memory_info().rss/2**30)
        p=model_prediction(model,items,q)
        np.savez_compressed(CACHE/f"{key}_weights.npz",**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        np.savez_compressed(CACHE/f"{key}_predictions.npz",indices=q,prediction=p)
        reload_model=SnapshotClassifier(len(state["keep"])+1,hidden=64,dropout=.15,kind="sage",residual=True,message_dropout=.2)
        with np.load(CACHE/f"{key}_weights.npz") as weights:
            reload_model.load_state_dict({k:torch.from_numpy(weights[k].copy()) for k in weights.files})
        # Re-evaluate all query nodes, including full final test queries.
        np.testing.assert_array_equal(model_prediction(reload_model,items,q),p)
    record=dict(key=key,profile=profile,fit_cutoff=cutoff,seed=seed,stage=stage,epochs=40,threads=2,
                fitting_labels=int(fitting.sum()),fit_feature_rows=state["fit_feature_rows"],
                features=len(state["keep"]),positive_multiplier=ratio,prediction_rows=len(q),
                fit_labels_sha256=hashlib.sha256(fitting_labels[fitting].tobytes()).hexdigest(),
                fit_indices_sha256=hashlib.sha256(np.flatnonzero(fitting).tobytes()).hexdigest(),
                transform_sha256=sha(CACHE/f"{profile}_c{cutoff}_transform.json"),
                weights_sha256=sha(CACHE/f"{key}_weights.npz"),
                seconds=time.perf_counter()-started,max_observed_rss_gib=rss,
                loss_by_epoch=losses,early_stopping=False,reload_all_query_predictions=True)
    fits.append(record);save("fit_log.json",fits);predictions[key]=(q.copy(),p)
    print(f"{key}: {record['seconds']:.1f}s, labels={record['fitting_labels']}, visible fit rows={record['fit_feature_rows']}",flush=True)
    del model,reload_model,optimizer,items;gc.collect()
    return p
