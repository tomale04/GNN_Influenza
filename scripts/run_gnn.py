"""Rolling evaluation of the mutation-tree GNN.

For each fold (test target T) and each config:
  1. Train an ensemble of SEEDS models on training pairs with CE loss to NN targets; every EVAL_EVERY epochs
     compute validation EMD of the ensemble (mean fitness); pick the best epoch count e*.
  2. Refit the ensemble on train+val for e* epochs; forecast the test pair; report EMD.
Configs: pooled vs per-subtype x mutation representation (bio / emb / both) x message passing (K=4) or none (K=0).
Usage: python3 run_gnn.py [config_name ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import forecast_framework as ff
from models.graph_data import build_graph, node_seq_ids
from models.gnn import MutationTreeGNN, batch_graphs, feature_stats

torch.set_num_threads(2)
# v2 (after v1 diagnostics: overfit within ~10 epochs, noisy val EMD): evaluate every epoch from epoch 0
# (epoch 0 == naive), early-stop on ensemble val CE, slower + more regularized training.
SEEDS, MAX_EPOCHS = int(os.environ.get("SEEDS", 5)), int(os.environ.get("MAX_EPOCHS", 150))
HP = dict(hidden=16, dropout=0.2, lr=1e-3, weight_decay=1e-2)
OUT_NAME = os.environ.get("OUT_NAME", "gnn_v2")
CONFIGS = {f"{pool}_{rep}_K{K}": dict(pool=pool, mut_repr=rep, layers=K)
           for pool in ("pooled", "persub") for rep, K in (("bio", 4), ("emb", 4), ("both", 4), ("bio", 0))}
CONFIGS.update({f"{pool}_bioseq_K{K}": dict(pool=pool, mut_repr="bio", layers=K, use_seq=True)
                for pool in ("pooled", "persub") for K in (4, 0)})


def make_data(keys, stats):
    gs = [dict(build_graph(s, t), seq=node_seq_ids(s, t)) for s, t in keys]
    b = batch_graphs(gs, stats)
    pairs = [ff.get_pair(s, t) for s, t in keys]
    for g, p in zip(gs, pairs):
        assert np.allclose(np.exp(g["log_x"].numpy()), p.current.freqs, atol=1e-6)
    b["y"] = torch.tensor(np.concatenate([p.target for p in pairs]), dtype=torch.float32)
    return b, pairs


def ce_loss(model, b):
    lp = model.log_probs(b)
    per = torch.zeros(b["n_graphs"]).index_add_(0, b["seg"], -(b["y"] * lp))
    return per.mean()  # each season pair weighted equally


def ensemble_emd(models, b, pairs):
    with torch.no_grad():
        for m in models: m.eval()
        f = torch.stack([m.fitness(b) for m in models]).mean(0)
        lp = models[0].log_probs(b, f)
        for m in models: m.train()
    out = []
    for gi, p in enumerate(pairs):
        pr = lp[b["seg"] == gi].exp().numpy()
        out.append(ff.emd(pr, p.future.freqs, p.D))
    return out, lp


def ensemble_val(models, b, pairs):
    emds, lp = ensemble_emd(models, b, pairs)
    ce = torch.zeros(b["n_graphs"]).index_add_(0, b["seg"], -(b["y"] * lp)).mean().item()
    return ce, float(np.mean(emds))


def train(cfg, train_keys, epochs, val=None, stats=None):
    bt, _ = make_data(train_keys, stats)
    models, opts = [], []
    for s in range(SEEDS):
        torch.manual_seed(s)
        m = MutationTreeGNN(cfg["mut_repr"], HP["hidden"], cfg["layers"], HP["dropout"], use_seq=cfg.get("use_seq", False))
        models.append(m); opts.append(torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]))
    curve = []
    for ep in range(0, epochs + 1):
        if val is not None:  # state after `ep` optimizer steps
            curve.append((ep, *ensemble_val(models, *val)))
        if ep == epochs: break
        for m, o in zip(models, opts):
            o.zero_grad(); ce_loss(m, bt).backward(); o.step()
    return models, curve


def run_config(name, cfg):
    rows, curves = [], []
    for fold in ff.folds():
        T = fold["test_target"]
        groups = [list(ff.SUBTYPES)] if cfg["pool"] == "pooled" else [[s] for s in ff.SUBTYPES]
        for subs in groups:
            tr = [k for k in fold["train"] if k[0] in subs]
            va = [k for k in fold["val"] if k[0] in subs]
            te = [k for k in fold["test"] if k[0] in subs]
            stats = feature_stats([build_graph(*k) for k in tr])
            val = make_data(va, stats)
            _, curve = train(cfg, tr, MAX_EPOCHS, val=val, stats=stats)
            best_ep, best_ce, best_val = min(curve, key=lambda c: c[1])  # select on val CE
            stats2 = feature_stats([build_graph(*k) for k in tr + va])
            models, _ = train(cfg, tr + va, best_ep, stats=stats2)
            bte, pte = make_data(te, stats2)
            emds, _ = ensemble_emd(models, bte, pte)
            for k, e in zip(te, emds):
                rows.append(dict(model=name, pair=ff.get_pair(*k).name, subtype=k[0], t=k[1], emd=e, best_epoch=best_ep, val_ce=best_ce, val_emd=best_val))
            curves.append(dict(T=T, subtypes=subs, curve=curve, best_epoch=best_ep))
            print(f"  {name} T={T} {'+'.join(subs)}: e*={best_ep} valCE={best_ce:.3f} (naive {curve[0][1]:.3f}) valEMD={best_val:.3f} test={[round(e, 3) for e in emds]}", flush=True)
    return rows, curves


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    out = os.path.join(ff.ROOT, "results", OUT_NAME); os.makedirs(out, exist_ok=True)
    for name in names:
        t0 = time.time()
        rows, curves = run_config(name, CONFIGS[name])
        pd.DataFrame(rows).to_csv(os.path.join(out, f"{name}.csv"), index=False)
        json.dump(curves, open(os.path.join(out, f"{name}_curves.json"), "w"))
        print(f"{name}: mean test EMD {np.mean([r['emd'] for r in rows]):.3f} ({time.time() - t0:.0f}s)", flush=True)
