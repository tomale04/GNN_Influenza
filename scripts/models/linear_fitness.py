"""Site-wise additive fitness model (ignores tree structure).

    f_i = sum_pos beta[pos, aa_i(pos)]                 (one coefficient per position-AA pair)
    xhat_i(t+1) = x_i(t) exp(f_i) / Z                  (Huddleston et al. 2020, Eq. 1; dt absorbed into beta)

beta = 0 recovers the naive forecast. Fit per subtype with L2 penalty lam * ||beta||^2, by
  - "ce":  multinomial cross-entropy to nearest-neighbour targets (convex; L-BFGS), or
  - "emd": exact 1-Wasserstein distance (LP dual potentials as (sub)gradients; Adam from beta=0).
Loss = mean over training season pairs (each pair weighted equally).
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import forecast_framework as ff
from scipy.optimize import linprog

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ features
class OneHot:
    def __init__(self, seasons):
        X = np.concatenate([s.X for s in seasons])
        L = X.shape[1]
        pairs = sorted({(p, a) for p in range(L) for a in np.unique(X[:, p])})
        self.index = {pa: k for k, pa in enumerate(pairs)}
        self.n = len(pairs)

    def __call__(self, X):
        F = np.zeros((X.shape[0], self.n))
        for p in range(X.shape[1]):
            for r, a in enumerate(X[:, p]):
                k = self.index.get((p, a))
                if k is not None: F[r, k] = 1.0
        return torch.tensor(F)


# ------------------------------------------------------------------ exact EMD with gradient
def emd_and_dual(p, q, D):
    m, n = D.shape
    A = np.zeros((m + n, m * n))
    for i in range(m): A[i, i * n:(i + 1) * n] = 1
    for j in range(n): A[m + j, j::n] = 1
    res = linprog(D.ravel(), A_eq=A[:-1], b_eq=np.concatenate([p, q])[:-1], bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return res.fun, res.eqlin.marginals[:m]  # d EMD / d p_i (up to an additive constant, killed by softmax)


class EMD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q, D):
        val, u = emd_and_dual(p.detach().numpy(), q, D)
        ctx.save_for_backward(torch.tensor(u))
        return torch.tensor(val)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * u, None, None


# ------------------------------------------------------------------ model
class LinearFitness:
    def __init__(self, loss="ce", lam=1.0, steps=300, lr=0.05):
        self.loss, self.lam, self.steps, self.lr = loss, lam, steps, lr

    def _prep(self, pair):
        return dict(F=self.feat(pair.current.X), logx=torch.tensor(np.log(pair.current.freqs)),
                    y=torch.tensor(pair.target), q=pair.future.freqs, D=pair.D)

    def _logp(self, d):
        return torch.log_softmax(d["logx"] + d["F"] @ self.beta, 0)

    def objective(self, data):
        tot = 0.0
        for d in data:
            lp = self._logp(d)
            tot = tot + (-(d["y"] * lp).sum() if self.loss == "ce" else EMD.apply(lp.exp(), d["q"], d["D"]))
        return tot / len(data) + self.lam * (self.beta ** 2).sum()

    def fit(self, pairs):
        self.feat = OneHot([p.current for p in pairs])
        data = [self._prep(p) for p in pairs]
        self.beta = torch.zeros(self.feat.n, requires_grad=True)
        if self.loss == "ce":
            opt = torch.optim.LBFGS([self.beta], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(); l = self.objective(data); l.backward(); return l
            opt.step(closure)
        else:
            opt = torch.optim.Adam([self.beta], lr=self.lr)
            best = (np.inf, None)
            for _ in range(self.steps):
                opt.zero_grad(); l = self.objective(data); l.backward(); opt.step()
                if l.item() < best[0]: best = (l.item(), self.beta.detach().clone())
            self.beta = best[1].requires_grad_(False)  # nonsmooth: keep best iterate
        self.beta = self.beta.detach()
        return self

    def predict(self, pair):
        with torch.no_grad():
            return self._logp(self._prep(pair)).exp().numpy()

    def coefficients(self):
        inv = {k: pa for pa, k in self.feat.index.items()}
        return {inv[k]: float(b) for k, b in enumerate(self.beta) if abs(b) > 1e-6}
