"""Regression + residual-blend GNN trained/validated on historical within-tree pairs.

Fold for test target T (test pair = actual snapshot, tree T-1 -> tree T):
  - training/validation data = historical views of tree T-1 at origins o <= T-2 (valid_origin), both subtypes;
  - validation origins = the K_VAL most recent origins, each fit on views with origin <= o-1;
  - regression: top-TOP configs by mean val EMD, forecasts averaged; final fit on all historical views;
  - GNN (residual on the regression ensemble, blended): (epoch, w) chosen by mean val EMD over all val pairs.
Usage: python3 -m hist.pipeline FAMILY [K ...]      (FAMILY in naive, lbi, mom, mlr_mom, nodereg, full)
"""
import os, sys, json, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd, torch
import forecast_framework as ff
from hist.views import get_view, view_features, view_graph, valid_origin
from models.gnn import MutationTreeGNN, batch_graphs, feature_stats

torch.set_num_threads(2)
K_VAL, TOP, PHI = int(os.environ.get("K_VAL", 6)), 5, 0.005
FAMILIES = {"naive": [], "lbi": ["lbi"], "mom": ["mom"], "mlr_mom": ["allele", "mom"],
            "nodereg": ["node", "mom", "lbi"], "full": ["node", "mom", "lbi", "allele"]}
TAU_PAIRS = [(0.25, 1.0), (0.5, 2.0), (0.25, 2.0)]
LAM_A, LAM_F, BASES = [0.01, 0.1, 1.0, 10.0], [1e-3, 1e-2, 1e-1], ["x", "pi"]
GNN_HP = dict(hidden=16, dropout=0.2, lr=1e-3, weight_decay=1e-2, seeds=5, max_epochs=150, ckpt=10)
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)
OUT = os.path.join(ff.ROOT, "results", f"hist_kval{K_VAL}")
# GNN node-input ablation: all 17 | base12 (drop momentum/LBI extras) | struct (has_leaves, is_candidate, is_root only)
NODE_MODE = os.environ.get("NODE_MODE", "all")
NODE_COLS = {"all": list(range(17)), "base12": list(range(12)), "struct": [6, 10, 11]}[NODE_MODE]


def gnn_graph(k):
    g = dict(view_graph(k)); g["x"] = g["x"][:, NODE_COLS]
    return g


# ------------------------------------------------------------------ data splits
def hist_keys(T, subtypes, max_origin):
    return [(s, T - 1, o) for s in subtypes for o in range(2005, max_origin + 1) if valid_origin(s, T - 1, o)]


def fold(T, subtypes):
    train_all = hist_keys(T, subtypes, T - 2)
    origins = sorted({k[2] for k in train_all}, reverse=True)
    val = []
    for o in origins:
        tr = [k for k in train_all if k[2] <= o - 1]
        if tr: val.append((o, tr, [k for k in train_all if k[2] == o]))
        if len(val) == K_VAL: break
    return dict(train=train_all, val=val, test=[(s, T - 1, T - 1) for s in subtypes])


def emd_of(p, key):
    v = get_view(*key)
    return ff.emd(p, v.ffreqs, v.D)


# ------------------------------------------------------------------ regression
def grid(blocks):
    g = []
    taus = TAU_PAIRS if "mom" in blocks else [(0.25, 1.0)]
    for tau, la, lf, base in itertools.product(taus, LAM_A if "allele" in blocks else [None], LAM_F if blocks else [None], BASES):
        if base == "pi" and "mom" not in blocks and tau != TAU_PAIRS[0]: continue
        g.append(dict(blocks=blocks, tau=tau, lam_allele=la, lam_feat=lf, base=base))
    return g


class Regression:
    def __init__(self, cfg): self.cfg = cfg

    def dense(self, key):
        vf = view_features(key); b = self.cfg["blocks"]; cols = []
        if "mom" in b: cols.append(vf["mom"](*self.cfg["tau"])[:, None])
        if "lbi" in b: cols += [vf["lbi_leaf"][:, None], vf["lbi_top"][:, None]]
        if "node" in b: cols.append(vf["x"][:, :12])
        D = np.concatenate(cols, 1) if cols else np.zeros((len(vf["cand"]), 0))
        return D[vf["cand"]]

    def base_log(self, key):
        v = get_view(*key)
        b = v.freqs if self.cfg["base"] == "x" else view_features(key)["pi"][self.cfg["tau"][0]][v.cand]
        return np.log(b / b.sum())

    def vocab_from(self, keys):
        vocab = {}
        for s in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == s]; acc = 0
            for k in ks:
                v = get_view(*k)
                acc = acc + (v.freqs[:, None, None] * (v.X[:, :, None] == np.arange(256, dtype=np.uint8)[None, None, :])).sum(0)
            acc = acc / len(ks)
            for p in range(acc.shape[0]):
                al = np.where(acc[p] >= PHI)[0]
                if len(al) >= 2:
                    for a in al: vocab[(s, p, int(a))] = len(vocab)
        return vocab

    def design(self, key):
        v = get_view(*key)
        D = (self.dense(key) - self.mu) / self.sd
        A = np.zeros((len(v.freqs), len(self.vocab)))
        for (s, p, a), j in self.vocab.items():
            if s == key[0]: A[:, j] = v.X[:, p] == a
        return np.concatenate([A, D], 1)

    def fit(self, keys):
        self.vocab = self.vocab_from(keys) if "allele" in self.cfg["blocks"] else {}
        Dall = np.concatenate([self.dense(k) for k in keys])
        self.mu = Dall.mean(0) if Dall.shape[1] else np.zeros(0)
        self.sd = np.where(Dall.std(0) > 1e-9, Dall.std(0), 1.0) if Dall.shape[1] else np.zeros(0)
        data = [(torch.tensor(self.design(k)), torch.tensor(self.base_log(k)), torch.tensor(get_view(*k).target)) for k in keys]
        na, d = len(self.vocab), data[0][0].shape[1]
        self.theta = torch.zeros(d, dtype=torch.float64)
        if d == 0: return self
        pen = torch.tensor([self.cfg["lam_allele"] or 0] * na + [self.cfg["lam_feat"]] * (d - na), dtype=torch.float64)
        th = self.theta.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([th], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = sum(-(y * torch.log_softmax(lb + Z @ th, 0)).sum() for Z, lb, y in data) / len(data) + (pen * th ** 2).sum()
            l.backward(); return l
        opt.step(closure)
        self.theta = th.detach()
        return self

    def predict(self, key):
        return torch.softmax(torch.tensor(self.base_log(key)) + torch.tensor(self.design(key)) @ self.theta, 0).numpy()


def reg_ensemble(cfgs, fit_keys, pred_keys):
    out = {k: 0.0 for k in pred_keys}
    for c in cfgs:
        m = Regression(c).fit(fit_keys)
        for k in pred_keys: out[k] = out[k] + m.predict(k) / len(cfgs)
    return out


def select_regression(fam, F):
    scored = []
    for cfg in grid(FAMILIES[fam]):
        e = [emd_of(Regression(cfg).fit(tr).predict(k), k) for _, tr, vk in F["val"] for k in vk]
        scored.append((float(np.mean(e)), cfg))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:TOP]], scored[0][0]


# ------------------------------------------------------------------ residual GNN + blend
def make_batch(keys, offsets, stats):
    gs = []
    for k in keys:
        g = gnn_graph(k); g["log_x"] = torch.tensor(np.log(np.clip(offsets[k], 1e-300, None)), dtype=torch.float32); gs.append(g)
    b = batch_graphs(gs, stats)
    b["y"] = torch.tensor(np.concatenate([get_view(*k).target for k in keys]), dtype=torch.float32)
    return b


def ce(lp, b):
    return torch.zeros(b["n_graphs"]).index_add_(0, b["seg"], -(b["y"] * lp))


def ens_probs(models, b):
    with torch.no_grad():
        for m in models: m.eval()
        f = torch.stack([m.fitness(b) for m in models]).mean(0); lp = models[0].log_probs(b, f)
        for m in models: m.train()
    return [lp[b["seg"] == gi].exp().numpy() for gi in range(b["n_graphs"])]


def train_gnn(K, bt, epochs, bv=None):
    H = GNN_HP; models, opts = [], []
    for sd in range(H["seeds"]):
        torch.manual_seed(sd)
        m = MutationTreeGNN("bio", H["hidden"], K, H["dropout"], n_node_feats=bt["x"].shape[1])
        models.append(m); opts.append(torch.optim.AdamW(m.parameters(), lr=H["lr"], weight_decay=H["weight_decay"]))
    ck = {}
    for ep in range(epochs + 1):
        if bv is not None and ep % H["ckpt"] == 0: ck[ep] = ens_probs(models, bv)
        if ep == epochs: break
        for m, o in zip(models, opts):
            o.zero_grad(); ce(m.log_probs(bt), bt).mean().backward(); o.step()
    return models, ck


def gnn_blend(K, cfgs, F):
    grid_ = {}
    for o, tr, vk in F["val"]:
        off = reg_ensemble(cfgs, tr, tr + vk)
        stats = feature_stats([gnn_graph(k) for k in tr])
        _, ck = train_gnn(K, make_batch(tr, off, stats), GNN_HP["max_epochs"], make_batch(vk, off, stats))
        for e, probs in ck.items():
            for w in (WEIGHTS if e > 0 else [0.0]):
                for k, pg in zip(vk, probs):
                    grid_.setdefault((e, float(w)), []).append(emd_of((1 - w) * off[k] + w * pg, k))
    score = {k: float(np.mean(v)) for k, v in grid_.items()}
    e_star, w_star = min(score, key=lambda k: (round(score[k], 6), k[1], k[0]))
    off = reg_ensemble(cfgs, F["train"], F["train"] + F["test"])
    stats = feature_stats([gnn_graph(k) for k in F["train"]])
    models, _ = train_gnn(K, make_batch(F["train"], off, stats), e_star)
    probs = ens_probs(models, make_batch(F["test"], off, stats))
    preds = {k: (1 - w_star) * off[k] + w_star * pg for k, pg in zip(F["test"], probs)}
    return preds, dict(e_star=e_star, w_star=w_star, val_reg=score[(0, 0.0)], val_best=score[(e_star, w_star)],
                       val_pairs_reg=grid_[(0, 0.0)], val_pairs_best=grid_[(e_star, w_star)],
                       val_pairs_w1={e: grid_[(e, 1.0)] for e in sorted({k[0] for k in grid_}) if e > 0},
                       val_w1_by_epoch={e: score.get((e, 1.0), score[(0, 0.0)]) for e in sorted({k[0] for k in score})})


# ------------------------------------------------------------------ main
def main(fam, Ks):
    os.makedirs(OUT, exist_ok=True)
    rows, logs = [], []
    subs = list(ff.SUBTYPES)
    for T in range(ff.FIRST_TEST_TARGET, ff.YEARS[-1] + 1):
        t0 = time.time(); F = fold(T, subs)
        if fam == "naive":
            cfgs, val = [dict(blocks=[], tau=(0.25, 1.0), lam_allele=None, lam_feat=None, base="x")], np.nan
        else:
            cfgs, val = select_regression(fam, F)
        reg = reg_ensemble(cfgs, F["train"], F["test"])
        for k in F["test"]:
            rows.append(dict(model=f"hist_{fam}", pair=f"{k[0]} {k[1]}->{k[1] + 1}", subtype=k[0], t=k[1], emd=emd_of(reg[k], k)))
        log = dict(T=T, n_train=len(F["train"]), val_origins=[o for o, _, _ in F["val"]], reg_val=val,
                   top=[{a: b for a, b in c.items() if a != "blocks"} for c in cfgs])
        for K in Ks:
            preds, info = gnn_blend(K, cfgs, F)
            for k in F["test"]:
                rows.append(dict(model=f"hist_{fam}+gnnblend_K{K}" + ("" if NODE_MODE == "all" else f"[{NODE_MODE}]"), pair=f"{k[0]} {k[1]}->{k[1] + 1}", subtype=k[0], t=k[1], emd=emd_of(preds[k], k), **{a: info[a] for a in ("e_star", "w_star")}))
            log[f"gnn_K{K}"] = info
        logs.append(log)
        cur = [r for r in rows if r["t"] == T - 1]
        print(f"T={T} train views {len(F['train'])} val origins {log['val_origins']} | " +
              " ".join(f"{r['model'].replace('hist_', '')}:{r['emd']:.2f}" for r in cur) +
              "".join(f" | K{K} e*={log[f'gnn_K{K}']['e_star']} w*={log[f'gnn_K{K}']['w_star']}" for K in Ks) + f" ({time.time() - t0:.0f}s)", flush=True)
    tag = fam + ("_gnn" + "".join(map(str, Ks)) if Ks else "") + ("" if NODE_MODE == "all" else f"_{NODE_MODE}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    json.dump(logs, open(os.path.join(OUT, f"{tag}_log.json"), "w"), indent=1, default=str)
    print(pd.DataFrame(rows).groupby("model").emd.mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], [int(k) for k in sys.argv[2:]])
