"""Residual GNN with a validation-chosen blend weight and EMD-based selection.

    p = (1 - w) * p_reg + w * p_gnn(e),   p_gnn(e) ∝ p_reg * exp(g_e)   (g = residual GNN after e epochs)

For each test target T: for each validation origin (same as run_regression / run_gnn_residual), refit the
regression ensemble on pairs before it, train the GNN ensemble, and record validation forecasts at
epochs 0, 10, ..., MAX_EPOCHS. (e*, w*) minimizes mean validation EMD over all validation pairs
(ties -> smaller w, then smaller e; w = 0 is exactly the regression). Then train on all pairs available at T
for e* epochs and blend with w*.
Usage: python3 run_gnn_blend.py REG_FAMILY [K ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import forecast_framework as ff
import selection as sel
from models.regression import Regression
from models.gnn import MutationTreeGNN, feature_stats
from run_regression import FAMILIES
import run_gnn_residual as rg

CKPT_EVERY = 10
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)
R = os.path.join(ff.ROOT, "results")


def gnn_probs(models, b):
    lp = rg.ens_logp(models, b)
    return [lp[b["seg"] == gi].exp().numpy() for gi in range(b["n_graphs"])]


def train_ckpt(K, bt, bv, epochs):
    """Train an ensemble; return {epoch: [val probs per val graph]} at checkpoints."""
    models, opts = [], []
    for sd in range(rg.SEEDS):
        torch.manual_seed(sd)
        m = MutationTreeGNN("bio", rg.HP["hidden"], K, rg.HP["dropout"], n_node_feats=bt["x"].shape[1])
        models.append(m); opts.append(torch.optim.AdamW(m.parameters(), lr=rg.HP["lr"], weight_decay=rg.HP["weight_decay"]))
    ck = {}
    for ep in range(epochs + 1):
        if ep % CKPT_EVERY == 0: ck[ep] = gnn_probs(models, bv)
        if ep == epochs: break
        for m, o in zip(models, opts):
            o.zero_grad(); rg.per_graph_ce(m.log_probs(bt), bt).mean().backward(); o.step()
    return ck


def main(fam, Ks):
    sels = json.load(open(os.path.join(R, "regression", f"{fam}_selection.json")))
    blocks = FAMILIES[fam.rsplit("_", 1)[0]]
    rows, logs = [], []
    for entry in sels:
        T, subs = entry["T"], entry["subtypes"]
        cfgs = [dict(c, blocks=blocks, tau=tuple(c["tau"])) for c in entry["top"]]
        test = [(s, T - 1) for s in subs]
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
            t0 = time.time()
            # grid[(e, w)] = list of val EMDs over all validation pairs
            grid = {}
            for fk, vk, off in folds:
                stats = feature_stats([rg.graph_with_extras(k) for k in fk])
                bt, bv = rg.make_batch(fk, off, stats), rg.make_batch(vk, off, stats)
                ck = train_ckpt(K, bt, bv, rg.MAX_EPOCHS)
                for e, probs in ck.items():
                    for w in (WEIGHTS if e > 0 else [0.0]):  # e = 0 is the regression for every w
                        for k, pg in zip(vk, probs):
                            pr = ff.get_pair(*k)
                            p = (1 - w) * off[k] + w * pg
                            grid.setdefault((e, float(w)), []).append(ff.emd(p, pr.future.freqs, pr.D))
            score = {k: float(np.mean(v)) for k, v in grid.items()}
            (e_star, w_star) = min(score, key=lambda k: (round(score[k], 6), k[1], k[0]))
            reg_val = score[(0, 0.0)]
            stats = feature_stats([rg.graph_with_extras(k) for k in fk_final])
            bt, bte = rg.make_batch(fk_final, off_final, stats), rg.make_batch(test, off_final, stats)
            models, _ = rg.train(K, bt, e_star, bt["x"].shape[1])
            probs = gnn_probs(models, bte)
            for k, pg in zip(test, probs):
                pr = ff.get_pair(*k)
                p = (1 - w_star) * off_final[k] + w_star * pg
                rows.append(dict(model=f"reg[{fam}]+gnnblend_K{K}", pair=pr.name, subtype=k[0], t=k[1],
                                 emd=ff.emd(p, pr.future.freqs, pr.D), e_star=e_star, w_star=w_star))
            # diagnostic: the val-EMD surface along w at e = MAX_EPOCHS and at e*
            logs.append(dict(T=T, K=K, e_star=e_star, w_star=w_star, val_emd_reg=reg_val, val_emd_best=score[(e_star, w_star)],
                             val_emd_w1_by_epoch={e: score.get((e, 1.0), score[(0, 0.0)]) for e in sorted({k[0] for k in score})},
                             val_emd_by_w_at_estar={w: score.get((e_star, float(w))) for w in WEIGHTS}))
            reg_te = [round(r["emd"], 3) for r in rows if r["model"] == f"reg[{fam}]" and r["t"] == T - 1]
            print(f"  T={T} K={K} e*={e_star} w*={w_star} valEMD reg={reg_val:.3f} best={score[(e_star, w_star)]:.3f} "
                  f"| test reg={reg_te} blend={[round(r['emd'], 3) for r in rows[-len(test):]]} ({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(R, "gnn_blend"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{fam}.csv"), index=False)
    json.dump(logs, open(os.path.join(out, f"{fam}_log.json"), "w"), indent=1)
    print(pd.DataFrame(rows).groupby("model").emd.mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], [int(k) for k in sys.argv[2:]] or [4, 0])
