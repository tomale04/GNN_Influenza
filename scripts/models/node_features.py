"""Extra per-node features from ~/Desktop/mlr_mom_K.pdf ideas + Nextstrain fitness metrics.

All quantities use only leaves dated <= snapshot T of tree t (leak-free).
- pi[tau]: exponentially discounted genotype share over a 5y lookback, weights exp(-(T - t_i)/tau)
  (mlr_mom_K Eq. 7), normalized over all nodes with leaves in the lookback.
- momentum(tau_f, tau_s) = log pi[tau_f] - log pi[tau_s], centered under pi[tau_f] over season-t
  genotypes (Eq. 10-11). Nodes without lookback leaves get 0 and a flag.
- lbi_leaf: mean Nextstrain LBI over the node's season-t leaves (else all its leaves); lbi_top: LBI at the
  Nextstrain node that defines the mutation-tree node. Both z-scored over season-t genotypes.
- ne_star (H3N2 2020+ only): mean over season-t leaves, z-scored; None where unavailable.
"""
import os, sys, functools, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import forecast_framework as ff
from auspice_io import load_dataset, iter_preorder
from build_mutation_trees import tip_date

TREES_DIR = os.path.join(ff.ROOT, "data", "nextstrain_trees")
LOOKBACK_YEARS = 5.0
TAUS = (0.25, 0.5, 1.0, 2.0)


def _attr(n, k):
    v = n.attrs.get(k)
    return v.get("value") if isinstance(v, dict) else v


@functools.lru_cache(maxsize=None)
def nextstrain_tables(subtype, year):
    ds = load_dataset(os.path.join(TREES_DIR, subtype, str(year)))
    leaf, node_lbi = {}, {}
    for n in iter_preorder(ds["tree"]):
        node_lbi[n.name] = _attr(n, "lbi")
        if n.is_tip:
            leaf[n.name] = (tip_date(n), _attr(n, "lbi"), _attr(n, "ne_star"))
    return leaf, node_lbi


def _z(v, mask):
    mu, sd = v[mask].mean(), v[mask].std()
    return (v - mu) / (sd if sd > 0 else 1.0)


@functools.lru_cache(maxsize=None)
def node_features(subtype, year):
    """Returns dict of arrays over ALL mutation-tree nodes of tree `year`."""
    mt = ff.load_tree(subtype, year)
    nodes = mt["nodes"]
    T = datetime.date.fromisoformat(mt["meta"]["snapshot_date"])
    leaf, node_lbi = nextstrain_tables(subtype, year)
    N = len(nodes)
    cand = np.array([n["n_recent"] > 0 for n in nodes])
    age = [np.array([(T - leaf[l][0]).days / 365.25 for l in n["leaves"]]) for n in nodes]
    out = {"cand": cand}
    for tau in TAUS:
        w = np.array([np.exp(-a[(a >= 0) & (a < LOOKBACK_YEARS)] / tau).sum() for a in age])
        out[f"pi_{tau}"] = w / w.sum()
    out["has_lookback"] = out[f"pi_{TAUS[0]}"] > 0

    def mean_over(n, k, recent_only):
        ls = n["recent_leaves"] if (recent_only and n["recent_leaves"]) else n["leaves"]
        vals = [leaf[l][k] for l in ls if leaf[l][k] is not None]
        return float(np.mean(vals)) if vals else np.nan

    lbi_leaf = np.array([mean_over(n, 1, True) for n in nodes])
    lbi_top = np.array([node_lbi.get(n["nextstrain_nodes"][0]) or 0.0 for n in nodes], float)
    out["has_leaves"] = ~np.isnan(lbi_leaf)
    out["lbi_leaf"] = np.nan_to_num(_z(np.nan_to_num(lbi_leaf), cand), nan=0.0) * out["has_leaves"]
    out["lbi_top"] = _z(lbi_top, cand)
    nes = np.array([mean_over(n, 2, True) for n in nodes])
    out["ne_star"] = _z(np.nan_to_num(nes), cand) if not np.isnan(nes[cand]).any() else None
    return out


def momentum(subtype, year, tau_f, tau_s):
    nf = node_features(subtype, year)
    pf, ps, ok = nf[f"pi_{tau_f}"], nf[f"pi_{tau_s}"], nf["has_lookback"]
    m = np.zeros(len(pf))
    m[ok] = np.log(pf[ok]) - np.log(ps[ok])
    c = nf["cand"]
    m[ok] -= (pf[c] / pf[c].sum() * m[c]).sum()
    return m


if __name__ == "__main__":
    for s in ff.SUBTYPES:
        for y in ff.YEARS:
            nf = node_features(s, y); c = nf["cand"]
            m = momentum(s, y, 0.25, 1.0)
            print(s, y, "cands", c.sum(), "| momentum(0.25,1) on cands: mean %.2f sd %.2f" % (m[c].mean(), m[c].std()),
                  "| corr(lbi_leaf, lbi_top) %.2f" % np.corrcoef(nf["lbi_leaf"][c], nf["lbi_top"][c])[0, 1],
                  "| ne_star", "yes" if nf["ne_star"] is not None else "no",
                  "| cands without lookback", int((c & ~nf["has_lookback"]).sum()))
