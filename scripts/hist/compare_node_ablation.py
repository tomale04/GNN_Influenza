"""Compare GNN node-input ablations (all 17 / base12 / struct) for the nodereg + K=4 blended GNN.

Validation (pooled over all folds' validation pairs, paired by pair):
  - 'selected': each variant at its own selected (e*, w*)  (selection-biased, equally for all variants)
  - 'w1@150' : GNN-corrected forecast (w=1) after 150 epochs, a fixed operating point
Test: the 10 actual snapshot pairs.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
from hist import pipeline as P

VARIANTS = {"all (17)": "", "base12": "_base12", "struct (3)": "_struct"}
rng = np.random.default_rng(0)


def boot(d):
    b = rng.choice(d, (10000, len(d))).mean(1)
    return f"{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}] (better in {(d < -1e-6).sum()}/{len(d)})"


val, test = {}, {}
for name, suf in VARIANTS.items():
    logs = json.load(open(os.path.join(P.OUT, f"nodereg_gnn4{suf}_log.json")))
    sel, w1, reg = [], [], []
    for L in logs:
        g = L["gnn_K4"]
        sel += g["val_pairs_best"]; reg += g["val_pairs_reg"]; w1 += g["val_pairs_w1"]["150"]
    val[name] = dict(selected=np.array(sel), w1_150=np.array(w1), reg=np.array(reg))
    df = pd.read_csv(os.path.join(P.OUT, f"nodereg_gnn4{suf}.csv"))
    test[name] = df[df.model.str.contains("gnnblend")].set_index("pair").emd
    test["regression"] = df[df.model == "hist_nodereg"].set_index("pair").emd

reg = val["all (17)"]["reg"]
print(f"Validation pairs: {len(reg)}; regression alone mean val EMD {reg.mean():.3f}\n")
rows = []
for name in VARIANTS:
    v = val[name]
    rows.append(dict(variant=name, val_selected=v["selected"].mean(), val_w1_150=v["w1_150"].mean(), test=test[name].mean(),
                     **{"val_selected vs all": boot(v["selected"] - val["all (17)"]["selected"]) if name != "all (17)" else "",
                        "val_w1@150 vs all": boot(v["w1_150"] - val["all (17)"]["w1_150"]) if name != "all (17)" else "",
                        "val_w1@150 vs regression": boot(v["w1_150"] - reg),
                        "test vs all": boot((test[name] - test["all (17)"]).values) if name != "all (17)" else "",
                        "test vs regression": boot((test[name] - test["regression"]).values)}))
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
S = pd.DataFrame(rows)
print(S.round(3).T.to_string())
S.to_csv(os.path.join(P.OUT, "node_ablation_comparison.csv"), index=False)
