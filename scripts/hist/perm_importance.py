"""Permutation importance of GNN node features (headline model: nodereg offset + K=4 GNN, all 17 node inputs).

For each test target T whose selected GNN epoch e* > 0 (from results/hist_kval6/nodereg_gnn4_log.json) and each of
its validation origins: train the GNN ensemble on the origin's training views for e* epochs, then for each node
feature shuffle that column across nodes within each validation tree (REPEATS times) and record the increase in
validation EMD of p_gnn (= the w=1 forecast; the headline selected w=1 whenever e*>0).
Output: results/hist_kval6/perm_importance.csv (one row per T, origin, val pair, feature, repeat).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd, torch
import forecast_framework as ff
from hist import pipeline as P
from hist.views import EXTRA_FEATS
from models.graph_data import NODE_FEATS
from models.gnn import feature_stats

REPEATS = 3
FEATS = NODE_FEATS + EXTRA_FEATS


def main():
    log = {L["T"]: L for L in json.load(open(os.path.join(P.OUT, "nodereg_gnn4_log.json")))}
    rng = np.random.default_rng(0)
    rows = []
    for T in range(ff.FIRST_TEST_TARGET, ff.YEARS[-1] + 1):
        e_star = log[T]["gnn_K4"]["e_star"]
        if e_star == 0:
            print(f"T={T}: e*=0 (GNN not used) - skipped", flush=True); continue
        F = P.fold(T, list(ff.SUBTYPES))
        cfgs = [dict(c, blocks=P.FAMILIES["nodereg"], tau=tuple(c["tau"])) for c in log[T]["top"]]
        for o, tr, vk in F["val"]:
            off = P.reg_ensemble(cfgs, tr, tr + vk)
            stats = feature_stats([P.gnn_graph(k) for k in tr])
            models, _ = P.train_gnn(4, P.make_batch(tr, off, stats), e_star)
            bv = P.make_batch(vk, off, stats)
            sizes = [P.gnn_graph(k)["n_nodes"] for k in vk]
            starts = np.cumsum([0] + sizes[:-1])
            base = [P.emd_of(p, k) for p, k in zip(P.ens_probs(models, bv), vk)]
            for j, fname in enumerate(FEATS):
                for r in range(REPEATS):
                    b2 = dict(bv); x = bv["x"].clone()
                    for st, n in zip(starts, sizes):
                        x[st:st + n, j] = x[st:st + n, j][torch.tensor(rng.permutation(n))]
                    b2["x"] = x
                    for k, p, b0 in zip(vk, P.ens_probs(models, b2), base):
                        rows.append(dict(T=T, origin=o, pair=f"{k[0]}@{o}", feature=fname, repeat=r, base_emd=b0, perm_emd=P.emd_of(p, k)))
            print(f"T={T} origin {o}: done", flush=True)
    df = pd.DataFrame(rows); df["delta"] = df.perm_emd - df.base_emd
    df.to_csv(os.path.join(P.OUT, "perm_importance.csv"), index=False)
    # summarize: average over repeats per (T, val pair, feature), then mean and bootstrap CI over val pairs
    g = df.groupby(["T", "pair", "feature"]).delta.mean().reset_index()
    out = []
    for f, d in g.groupby("feature"):
        v = d.delta.values; b = rng.choice(v, (10000, len(v))).mean(1)
        out.append(dict(feature=f, mean_increase=v.mean(), ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_val_pairs=len(v)))
    S = pd.DataFrame(out).sort_values("mean_increase", ascending=False)
    S.to_csv(os.path.join(P.OUT, "perm_importance_summary.csv"), index=False)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
