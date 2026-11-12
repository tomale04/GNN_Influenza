"""Forecasting framework: season t -> season t+1 HA1+HA2 genotype distributions.

Setup (following Huddleston et al. 2020, eLife 9:e60067):
- Season t = samples collected Oct 1 (t-1) .. Apr 1 (t), i.e. the `n_recent` leaves of mutation tree t.
- A forecast assigns a predicted season-(t+1) frequency to each season-t genotype (mutation-tree node
  of tree t with n_recent > 0). No new mutations are forecast.
- Truth = observed season-(t+1) genotype frequencies (n_recent leaves of mutation tree t+1).
- Eval metric = earth mover's (1-Wasserstein) distance between predicted and observed populations
  with Hamming distance over HA1+HA2 amino acids as ground distance (units: amino acids).
- Training surrogate: map each season-(t+1) leaf to its nearest season-t genotype(s) by Hamming
  distance (ties split equally) -> target distribution y over season-t genotypes; fit with
  multinomial cross-entropy  -sum_i y_i log p_i. Predicting p = y exactly attains the EMD lower bound
  (sum_j q_j min_i d_ij), so y is the oracle forecast.

Rolling evaluation: a forecast of season T may use all data through season T-1 (mutation trees <= T-1).
Fold for test target T: test pair (T-1 -> T), validation pair (T-2 -> T-1), training pairs with
targets <= T-2. Pairs from both subtypes are available in every fold.
"""
import os, json, functools
import numpy as np
from scipy.optimize import linprog

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MT_DIR = os.path.join(ROOT, "data", "mutation_trees")
SUBTYPES = ("h3n2", "h1n1pdm")
YEARS = tuple(range(2019, 2027))
FIRST_TEST_TARGET = 2022  # needs >=1 training pair and 1 validation pair before it


# ---------------------------------------------------------------- data
@functools.lru_cache(maxsize=None)
def load_tree(subtype, year):
    return json.load(open(os.path.join(MT_DIR, subtype, f"{year}.json")))


def encode(seqs):
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)


class Season:
    """Genotypes observed in a season: mutation-tree nodes of tree `year` with n_recent > 0."""

    def __init__(self, subtype, year):
        t = load_tree(subtype, year)
        nodes = [n for n in t["nodes"] if n["n_recent"] > 0]
        self.subtype, self.year = subtype, year
        self.node_ids = np.array([n["id"] for n in nodes])
        self.seqs = [n["HA1"] + n["HA2"] for n in nodes]
        self.X = encode(self.seqs)
        self.counts = np.array([n["n_recent"] for n in nodes], dtype=float)
        self.freqs = self.counts / self.counts.sum()
        self.tree = t  # full mutation tree, for building model features


def hamming(A, B):
    return (A[:, None, :] != B[None, :, :]).sum(-1).astype(float)


class ForecastPair:
    """One forecasting problem: season t (inputs) -> season t+1 (truth)."""

    def __init__(self, subtype, t):
        self.subtype, self.t = subtype, t
        self.current = Season(subtype, t)
        self.future = Season(subtype, t + 1)
        self.D = hamming(self.current.X, self.future.X)  # [n_current, n_future]
        self.target = nn_targets(self.D, self.future.freqs)

    @property
    def name(self):
        return f"{self.subtype} {self.t}->{self.t + 1}"


@functools.lru_cache(maxsize=None)
def get_pair(subtype, t):
    return ForecastPair(subtype, t)


# ---------------------------------------------------------------- objectives
def emd(p, q, D):
    """Exact 1-Wasserstein distance between weights p (rows of D) and q (cols of D)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m, n = D.shape
    A_eq = np.zeros((m + n, m * n))
    for i in range(m): A_eq[i, i * n:(i + 1) * n] = 1
    for j in range(n): A_eq[m + j, j::n] = 1
    res = linprog(D.ravel(), A_eq=A_eq[:-1], b_eq=np.concatenate([p, q])[:-1], bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def nn_targets(D, q):
    """Map each future genotype's mass to its nearest current genotype(s); ties split equally."""
    nearest = D == D.min(axis=0, keepdims=True)
    return (nearest / nearest.sum(axis=0, keepdims=True) * q[None, :]).sum(axis=1)


def emd_lower_bound(D, q):
    return float((D.min(axis=0) * q).sum())


def cross_entropy(pred, target, eps=1e-12):
    """Multinomial training loss -sum_i y_i log p_i (numpy; mirror this in torch for models)."""
    pred = np.asarray(pred, float); pred = pred / pred.sum()
    return float(-(target * np.log(pred + eps)).sum())


# ---------------------------------------------------------------- splits
def folds(subtypes=SUBTYPES):
    """Rolling folds. Each yields dict(test_target=T, train=[pairs], val=[pairs], test=[pairs])."""
    out = []
    for T in range(FIRST_TEST_TARGET, YEARS[-1] + 1):
        out.append(dict(
            test_target=T,
            train=[(s, t) for s in subtypes for t in YEARS if t + 1 <= T - 2],
            val=[(s, T - 2) for s in subtypes],
            test=[(s, T - 1) for s in subtypes],
        ))
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(predict, pairs):
    """predict(pair) -> nonnegative weights over pair.current genotypes. Returns list of metric dicts."""
    rows = []
    for s, t in pairs:
        pr = get_pair(s, t)
        p = np.asarray(predict(pr), float)
        assert p.shape == pr.current.freqs.shape and (p >= 0).all() and p.sum() > 0
        rows.append(dict(pair=pr.name, subtype=s, t=t, emd=emd(p, pr.future.freqs, pr.D),
                         ce=cross_entropy(p, pr.target), lower_bound=emd_lower_bound(pr.D, pr.future.freqs),
                         n_current=len(p), n_future=len(pr.future.freqs)))
    return rows


# reference forecasts (no learning)
def naive(pair):
    """Predicted season t+1 frequency = observed season t frequency (Huddleston 2020 'naive' model)."""
    return pair.current.freqs


def uniform(pair):
    return np.ones(len(pair.current.freqs))


def oracle(pair):
    """Uses the future (not a valid forecast): attains the EMD lower bound."""
    return pair.target
