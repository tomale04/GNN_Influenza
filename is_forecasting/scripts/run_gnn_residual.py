"""Does the GNN add signal beyond the regression on node features?

Model:  p ∝ exp( log p_reg + g ),  g = GNN(mutation tree t) with zero-init readout, so epoch 0 == regression.
p_reg = averaged forecast of the regression family's top-5 configs (as selected in run_regression.py for
that test target), refit on exactly the pairs available at each origin.
GNN node features = original 12 + momentum(0.25,1), momentum(0.5,2), lbi_leaf, lbi_top, has_lookback.
Epochs chosen by mean validation-CE improvement over the same validation origins as the regression; e*=0 allowed.
Variants: K=4 (tree message passing) vs K=0 (per-node nonlinearity only).
Note: offsets on training pairs are in-sample regression fits (no cross-fitting).
Usage: python3 run_gnn_residual.py REG_FAMILY [K ...]     e.g. full_pooled 4 0
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import forecast_framework as ff
import selection as sel
from models.regression import Regression
from models.node_features import node_features, momentum
from models.graph_data import build_graph
from models.gnn import MutationTreeGNN, batch_graphs, feature_stats
from run_regression import FAMILIES

torch.set_num_threads(2)
SEEDS, MAX_EPOCHS = 5, 150
HP = dict(hidden=16, dropout=0.2, lr=1e-3, weight_decay=1e-2)
R = os.path.join(ff.ROOT, "results")


def graph_with_extras(key):
    s, t = key
    g = dict(build_graph(s, t)); nf = node_features(s, t)
    extra = np.stack([momentum(s, t, 0.25, 1.0), momentum(s, t, 0.5, 2.0), nf["lbi_leaf"], nf["lbi_top"], nf["has_lookback"].astype(float)], 1)
    g["x"] = torch.cat([g["x"], torch.tensor(extra, dtype=torch.float32)], 1)
    return g


def make_batch(keys, offsets, stats):
    gs = [graph_with_extras(k) for k in keys]
    for g, k in zip(gs, keys):
        g["log_x"] = torch.tensor(np.log(np.clip(offsets[k], 1e-300, None)), dtype=torch.float32)
    b = batch_graphs(gs, stats)
    b["y"] = torch.tensor(np.concatenate([ff.get_pair(*k).target for k in keys]), dtype=torch.float32)
    return b


def per_graph_ce(lp, b):
    return torch.zeros(b["n_graphs"]).index_add_(0, b["seg"], -(b["y"] * lp))


def ens_logp(models, b):
    with torch.no_grad():
        for m in models: m.eval()
        f = torch.stack([m.fitness(b) for m in models]).mean(0)
        lp = models[0].log_probs(b, f)
        for m in models: m.train()
    return lp


def train(K, bt, epochs, n_feats, bv=None):
    models, opts = [], []
    for sd in range(SEEDS):
        torch.manual_seed(sd)
        m = MutationTreeGNN("bio", HP["hidden"], K, HP["dropout"], n_node_feats=n_feats)
        models.append(m); opts.append(torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]))
    curve = []
    for ep in range(epochs + 1):
        if bv is not None: curve.append(per_graph_ce(ens_logp(models, bv), bv).mean().item())
        if ep == epochs: break
        for m, o in zip(models, opts):
            o.zero_grad(); per_graph_ce(m.log_probs(bt), bt).mean().backward(); o.step()
    return models, np.array(curve)


def main(fam, Ks):
    sels = json.load(open(os.path.join(R, "regression", f"{fam}_selection.json")))
    blocks = FAMILIES[fam.rsplit("_", 1)[0]]
    rows, logs = [], []
    for entry in sels:
        T, subs = entry["T"], entry["subtypes"]
        cfgs = [dict(c, blocks=blocks, tau=tuple(c["tau"])) for c in entry["top"]]
        test = [(s, T - 1) for s in subs]
        # validation folds: regression refit on pairs before each origin
        folds = []
        for tv, vk in sel.val_origins(T, subs):
            fk = sel.train_keys(subs, tv)
            folds.append((fk, vk, sel.ensemble_predict(Regression, cfgs, fk, fk + vk)))
        fk_final = sel.train_keys(subs, T - 1)
        off_final = sel.ensemble_predict(Regression, cfgs, fk_final, fk_final + test)
        for k in test:
            pr = ff.get_pair(*k)
            rows.append(dict(model=f"reg[{fam}]", pair=pr.name, subtype=k[0], t=k[1], emd=ff.emd(off_final[k], pr.future.freqs, pr.D)))
        for K in Ks:
            t0 = time.time(); curves = []
            for fk, vk, off in folds:
                stats = feature_stats([graph_with_extras(k) for k in fk])
                bt, bv = make_batch(fk, off, stats), make_batch(vk, off, stats)
                n_feats = bt["x"].shape[1]
                _, c = train(K, bt, MAX_EPOCHS, n_feats, bv)
                curves.append(c - c[0])
            mean_curve = np.mean(curves, 0); e_star = int(np.argmin(mean_curve))
            stats = feature_stats([graph_with_extras(k) for k in fk_final])
            bt, bte = make_batch(fk_final, off_final, stats), make_batch(test, off_final, stats)
            models, _ = train(K, bt, e_star, bt["x"].shape[1])
            lp = ens_logp(models, bte)
            for gi, k in enumerate(test):
                pr = ff.get_pair(*k)
                p = lp[bte["seg"] == gi].exp().numpy()
                rows.append(dict(model=f"reg[{fam}]+gnn_K{K}", pair=pr.name, subtype=k[0], t=k[1], emd=ff.emd(p, pr.future.freqs, pr.D), e_star=e_star))
            logs.append(dict(T=T, K=K, e_star=e_star, val_dCE_at_estar=float(mean_curve[e_star]), n_folds=len(folds)))
            print(f"  T={T} K={K} e*={e_star} valΔCE={mean_curve[e_star]:+.4f} reg={[round(r['emd'], 3) for r in rows if r['model'] == f'reg[{fam}]' and r['t'] == T - 1]} "
                  f"+gnn={[round(r['emd'], 3) for r in rows[-len(test):]]} ({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(R, "gnn_residual"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{fam}.csv"), index=False)
    json.dump(logs, open(os.path.join(out, f"{fam}_log.json"), "w"), indent=1)
    df = pd.DataFrame(rows)
    print(df.groupby("model").emd.mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], [int(k) for k in sys.argv[2:]] or [4, 0])
