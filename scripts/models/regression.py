"""Multinomial-logit fitness regression on node features (mlr+mom family, mlr_mom_K.pdf Sec. 3-5).

    p_v  ∝  b_v exp( beta^T S_v + gamma^T z_v )        over season-t genotypes v
    b    = current-season frequency x ("x") or discounted share pi[tau_f] ("pi")
    S    = filtered allele one-hot (site-allele pairs with training-season frequency >= PHI at sites with
           >= 2 such alleles; subtype-specific)
    z    = dense features from blocks: mom (momentum), lbi (lbi_leaf, lbi_top), nestar, node (GNN node features)
Fit: mean over training pairs of CE to nearest-neighbour targets + lam_allele ||beta||^2 + lam_feat ||gamma||^2
(anisotropic penalty), convex -> L-BFGS. Dense features standardized with training statistics.
"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import forecast_framework as ff
from models.node_features import node_features, momentum
from models.graph_data import build_graph

PHI = 0.005


def available(cfg, key):
    s, t = key
    if "nestar" in cfg["blocks"]:
        return node_features(s, t)["ne_star"] is not None
    return True


def dense_block(cfg, key):
    """Dense features over ALL nodes of tree t (rows = nodes)."""
    s, t = key
    nf = node_features(s, t)
    cols = []
    if "mom" in cfg["blocks"]:
        cols.append(momentum(s, t, *cfg["tau"])[:, None])
    if "lbi" in cfg["blocks"]:
        cols += [nf["lbi_leaf"][:, None], nf["lbi_top"][:, None]]
    if "nestar" in cfg["blocks"]:
        cols.append(nf["ne_star"][:, None])
    if "node" in cfg["blocks"]:
        cols.append(build_graph(s, t)["x"].numpy().astype(float))
    return np.concatenate(cols, 1) if cols else np.zeros((len(nf["cand"]), 0))


def base_log(cfg, key):
    s, t = key
    nf = node_features(s, t); c = nf["cand"]
    b = ff.get_pair(s, t).current.freqs if cfg.get("base", "x") == "x" else nf[f"pi_{cfg['tau'][0]}"][c]
    return np.log(b / b.sum())


class Regression:
    def __init__(self, cfg):
        self.cfg = cfg

    # ---- design
    def _allele_vocab(self, keys):
        vocab = {}
        for s in sorted({k[0] for k in keys}):
            acc = None
            ks = [k for k in keys if k[0] == s]
            for k in ks:
                cur = ff.get_pair(*k).current
                onehot = cur.X[:, :, None] == np.arange(256, dtype=np.uint8)[None, None, :]  # [n, L, 256]
                f = (cur.freqs[:, None, None] * onehot).sum(0)
                acc = f if acc is None else acc + f
            acc /= len(ks)
            for p in range(acc.shape[0]):
                al = np.where(acc[p] >= PHI)[0]
                if len(al) >= 2:
                    for a in al: vocab[(s, p, int(a))] = len(vocab)
        return vocab

    def _design(self, key):
        s, t = key
        cur = ff.get_pair(s, t).current
        c = node_features(s, t)["cand"]
        D = dense_block(self.cfg, key)[c]
        D = (D - self.mu) / self.sd
        A = np.zeros((len(cur.freqs), len(self.vocab)))
        for (ss, p, a), j in self.vocab.items():
            if ss == s: A[:, j] = cur.X[:, p] == a
        return np.concatenate([A, D], 1)

    def fit(self, keys):
        self.vocab = self._allele_vocab(keys) if "allele" in self.cfg["blocks"] else {}
        Ds = [dense_block(self.cfg, k)[node_features(*k)["cand"]] for k in keys]
        Dall = np.concatenate(Ds) if Ds[0].shape[1] else np.zeros((1, 0))
        self.mu, self.sd = Dall.mean(0), np.where(Dall.std(0) > 1e-9, Dall.std(0), 1.0)
        data = [(torch.tensor(self._design(k)), torch.tensor(base_log(self.cfg, k)), torch.tensor(ff.get_pair(*k).target)) for k in keys]
        na = len(self.vocab); d = data[0][0].shape[1]
        pen = torch.tensor([self.cfg.get("lam_allele", 1.0)] * na + [self.cfg.get("lam_feat", 1e-2)] * (d - na))
        self.theta = torch.zeros(d, dtype=torch.float64, requires_grad=True)
        if d == 0:
            self.theta = self.theta.detach(); return self
        opt = torch.optim.LBFGS([self.theta], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = sum(-(y * torch.log_softmax(lb + Z @ self.theta, 0)).sum() for Z, lb, y in data) / len(data) + (pen * self.theta ** 2).sum()
            l.backward(); return l
        opt.step(closure)
        self.theta = self.theta.detach()
        return self

    def log_predict(self, key):
        Z = torch.tensor(self._design(key)); lb = torch.tensor(base_log(self.cfg, key))
        return torch.log_softmax(lb + Z @ self.theta, 0).numpy()

    def predict(self, key):
        return np.exp(self.log_predict(key))

    def coefficients(self):
        names = [f"{s}:{'HA1' if p < (329 if s == 'h3n2' else 327) else 'HA2'}{p + 1}{chr(a)}" for (s, p, a) in self.vocab]
        return dict(zip(names + [f"dense{i}" for i in range(len(self.theta) - len(names))], self.theta.numpy().tolist()))
