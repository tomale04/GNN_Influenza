"""Tree views: a mutation tree as it looked at an origin date, for historical training/validation pairs.

A view key is (subtype, tree_year, origin):
  - origin == tree_year: the actual snapshot (cutoff = snapshot date); future season = n_recent of tree tree_year+1.
  - origin <  tree_year: tree tree_year pruned to leaves dated <= Apr 1 of `origin`; future season = tree
    tree_year's own leaves dated Oct 1 (origin) .. Apr 1 (origin+1).
Season `origin` (current) = leaves dated Oct 1 (origin-1) .. cutoff.
Pruning keeps a node iff it has a retained leaf in its subtree. LBI is recomputed on the pruned time tree
(augur algorithm, tau=0.5, window=0.5), so nothing dated after the cutoff enters any input feature.
Residual leakage for historical views: topology, internal-node genotypes and dates of tree tree_year were inferred
with tips after the cutoff. Test pairs always use actual snapshots and are unaffected.
"""
import os, sys, functools, datetime, collections, math
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import forecast_framework as ff
from auspice_io import load_dataset, iter_preorder
from build_mutation_trees import tip_date
from lbi import compute_lbi
from models.graph_data import AAS, AA_ID, KD, CHARGE, VOLUME, BLOSUM, SITES, NODE_FEATS, MUT_FEATS

TREES_DIR = os.path.join(ff.ROOT, "data", "nextstrain_trees")
LBI_TAU, LBI_WINDOW = 0.5, 0.5
LOOKBACK_YEARS = 5.0
TAUS = (0.25, 0.5, 1.0, 2.0)
MIN_SEASON_LEAVES = 30
EXTRA_FEATS = ["mom_0.25_1", "mom_0.5_2", "lbi_leaf", "lbi_top", "has_lookback"]


def season_bounds(origin):
    return datetime.date(origin - 1, 10, 1), datetime.date(origin, 4, 1)


@functools.lru_cache(maxsize=None)
def nextstrain(subtype, year):
    ds = load_dataset(os.path.join(TREES_DIR, subtype, str(year)))
    dates = {n.name: tip_date(n) for n in iter_preorder(ds["tree"]) if n.is_tip}
    return ds["tree"], dates


def to_decimal(d):
    y = d.year; n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    return y + (d.timetuple().tm_yday - 0.5) / n


@functools.lru_cache(maxsize=None)
def season_counts(subtype, year):
    _, dates = nextstrain(subtype, year)
    c = collections.Counter()
    for d in dates.values():
        if d.month >= 10: c[d.year + 1] += 1
        elif d.month < 4 or (d.month == 4 and d.day == 1): c[d.year] += 1
    return c


def valid_origin(subtype, tree_year, origin):
    """Historical origin usable: seasons origin-1, origin, origin+1 all well sampled in the tree."""
    if origin == tree_year: return True
    c = season_counts(subtype, tree_year)
    return origin + 1 <= tree_year and all(c[s] >= MIN_SEASON_LEAVES for s in (origin - 1, origin, origin + 1))


class View:
    def __init__(self, subtype, tree_year, origin):
        self.key = (subtype, tree_year, origin)
        self.subtype, self.tree_year, self.origin = subtype, tree_year, origin
        mt = ff.load_tree(subtype, tree_year)
        snap = datetime.date.fromisoformat(mt["meta"]["snapshot_date"])
        self.cutoff = snap if origin == tree_year else datetime.date(origin, 4, 1)
        root, dates = nextstrain(subtype, tree_year)
        self.dates = dates
        s0, _ = season_bounds(origin)
        keep = {l for l, d in dates.items() if d <= self.cutoff}
        full = mt["nodes"]
        # retention (nodes are in preorder: parents before children)
        has = [any(l in keep for l in n["leaves"]) for n in full]
        for i in range(len(full) - 1, 0, -1):
            if has[i]: has[full[i]["parent"]] = True
        idx = {old: new for new, old in enumerate(i for i in range(len(full)) if has[i])}
        self.nodes = []
        for old, n in enumerate(full):
            if not has[old]: continue
            leaves = [l for l in n["leaves"] if l in keep]
            self.nodes.append(dict(parent=idx[n["parent"]] if n["parent"] is not None else None, mutations=n["mutations"],
                                   seq=n["HA1"] + n["HA2"], depth=len(n["mutations_from_reference"]), leaves=leaves,
                                   recent=[l for l in leaves if s0 <= dates[l] <= self.cutoff], top=n["nextstrain_nodes"][0]))
        self.n_recent = np.array([len(n["recent"]) for n in self.nodes])
        self.cand = np.where(self.n_recent > 0)[0]
        self.X = ff.encode([self.nodes[i]["seq"] for i in self.cand])
        self.freqs = self.n_recent[self.cand] / self.n_recent[self.cand].sum()
        cut_dec = max(to_decimal(dates[l]) for l in keep)
        self.lbi = compute_lbi(root, cut_dec, LBI_TAU, LBI_WINDOW, keep=keep)
        # future season
        if origin == tree_year:
            fut = ff.Season(subtype, tree_year + 1)
            self.fX, self.ffreqs = fut.X, fut.freqs
        else:
            f0, f1 = season_bounds(origin + 1)
            cnt = collections.Counter()
            for n in full:
                k = sum(f0 <= dates[l] <= f1 for l in n["leaves"])
                if k: cnt[n["HA1"] + n["HA2"]] += k
            seqs = sorted(cnt); w = np.array([cnt[s] for s in seqs], float)
            self.fX, self.ffreqs = ff.encode(seqs), w / w.sum()
        self.D = ff.hamming(self.X, self.fX)
        self.target = ff.nn_targets(self.D, self.ffreqs)

    @property
    def name(self):
        return f"{self.subtype} tree{self.tree_year}@{self.origin}->{self.origin + 1}"


@functools.lru_cache(maxsize=None)
def get_view(subtype, tree_year, origin):
    return View(subtype, tree_year, origin)


# ------------------------------------------------------------------ node features
def _z(v, mask):
    sd = v[mask].std()
    return (v - v[mask].mean()) / (sd if sd > 0 else 1.0)


@functools.lru_cache(maxsize=None)
def view_features(key):
    """Returns dict: x [N, len(NODE_FEATS)+len(EXTRA_FEATS)] node features, plus named arrays."""
    v = get_view(*key)
    N = len(v.nodes); cand_mask = v.n_recent > 0
    c0 = v.cutoff
    w = [season_bounds(v.origin),
         (datetime.date(v.origin - 1, 4, 1), datetime.date(v.origin - 1, 10, 1) - datetime.timedelta(days=1)),
         (datetime.date(v.origin - 2, 10, 1), datetime.date(v.origin - 1, 4, 1) - datetime.timedelta(days=1))]
    w[0] = (w[0][0], c0)
    cnt = np.zeros((N, 3)); first = [None] * N; last = [None] * N
    for i, n in enumerate(v.nodes):
        ds_ = [v.dates[l] for l in n["leaves"]]
        for k, (a, b) in enumerate(w):
            cnt[i, k] = sum(a <= d <= b for d in ds_)
        if ds_: first[i], last[i] = min(ds_), max(ds_)
    assert (cnt[:, 0] == v.n_recent).all()
    tot = cnt.sum(0); tot[tot == 0] = 1; freq = cnt / tot
    depth = np.array([n["depth"] for n in v.nodes], float)
    X = np.zeros((N, len(NODE_FEATS)))
    X[:, 0:3] = freq; X[:, 3:6] = np.log1p(cnt)
    for i in range(N):
        if first[i]:
            X[i, 6] = 1; X[i, 7] = (c0 - first[i]).days / 365.25; X[i, 8] = (c0 - last[i]).days / 365.25
    X[:, 9] = (depth - (depth[cand_mask] * freq[cand_mask, 0]).sum()) / 10
    X[:, 10] = cand_mask; X[0, 11] = 1
    # discounted shares + momentum
    age = [np.array([(c0 - v.dates[l]).days / 365.25 for l in n["leaves"]]) for n in v.nodes]
    pi = {}
    for tau in TAUS:
        ww = np.array([np.exp(-a[(a >= 0) & (a < LOOKBACK_YEARS)] / tau).sum() for a in age])
        pi[tau] = ww / ww.sum()
    ok = pi[TAUS[0]] > 0

    def mom(tf, ts):
        m = np.zeros(N); m[ok] = np.log(pi[tf][ok]) - np.log(pi[ts][ok])
        pf = pi[tf][cand_mask]; m[ok] -= (pf / pf.sum() * m[cand_mask]).sum()
        return m
    lbi_leaf = np.zeros(N); has_l = np.zeros(N, bool)
    for i, n in enumerate(v.nodes):
        ls = n["recent"] or n["leaves"]
        if ls: lbi_leaf[i] = np.mean([v.lbi[l] for l in ls]); has_l[i] = True
    lbi_leaf = _z(lbi_leaf, cand_mask) * has_l
    lbi_top = _z(np.array([v.lbi[n["top"]] for n in v.nodes]), cand_mask)
    extra = np.stack([mom(0.25, 1.0), mom(0.5, 2.0), lbi_leaf, lbi_top, ok.astype(float)], 1)
    return dict(x=np.concatenate([X, extra], 1), pi=pi, mom=mom, lbi_leaf=lbi_leaf, lbi_top=lbi_top, cand=cand_mask)


@functools.lru_cache(maxsize=None)
def view_graph(key):
    """GNN graph dict compatible with models.gnn.batch_graphs."""
    v = get_view(*key); vf = view_features(key)
    subtype = v.subtype; N = len(v.nodes); L1 = 329 if subtype == "h3n2" else 327
    w = [season_bounds(v.origin), None, None]
    feats = vf["x"]
    # residue frequencies per window from node window frequencies (cols 0..2)
    res = [collections.defaultdict(collections.Counter) for _ in range(3)]
    for i, n in enumerate(v.nodes):
        for k in range(3):
            if feats[i, k] > 0:
                for p, a in enumerate(n["seq"]): res[k][p][a] += feats[i, k]
    epi = set(SITES[subtype]["epitope"]); koel = set(SITES[subtype]["koel"])
    homo = collections.Counter(m for n in v.nodes[1:] for m in n["mutations"])
    anc = {0: collections.defaultdict(set)}
    for p, a in enumerate(v.nodes[0]["seq"]): anc[0][p].add(a)
    MF, ME, MP, MFr, MT, EC, EP = [], [], [], [], [], [], []
    for i, n in enumerate(v.nodes):
        if n["parent"] is None: continue
        par = n["parent"]; e = len(EC); EC.append(i); EP.append(par)
        st = collections.defaultdict(set, {k: set(s) for k, s in anc[par].items()})
        for m in n["mutations"]:
            gene, rest = m.split(":"); f, pos, t = rest[0], int(rest[1:-1]), rest[-1]
            gp = (pos - 1) if gene == "HA1" else (L1 + pos - 1)
            fv = [gene == "HA1", gene == "HA1" and pos in epi, gene == "HA1" and pos in koel,
                  BLOSUM[f][t] / 4 if f in AAS and t in AAS else 0, (KD.get(t, 0) - KD.get(f, 0)) / 4.5,
                  CHARGE.get(t, 0) - CHARGE.get(f, 0), (VOLUME.get(t, 120) - VOLUME.get(f, 120)) / 100,
                  t in anc[par][gp], np.log1p(homo[m] - 1)]
            fv += [res[k][gp][t] for k in range(3)] + [res[k][gp][f] for k in range(3)]
            MF.append(fv); ME.append(e); MP.append(gp); MFr.append(AA_ID.get(f, 0)); MT.append(AA_ID.get(t, 0))
            st[gp].add(t)
        anc[i] = st
    cand = v.cand
    return dict(subtype=subtype, n_nodes=N, x=torch.tensor(feats, dtype=torch.float32),
                edge_child=torch.tensor(EC), edge_parent=torch.tensor(EP),
                n_mut_per_edge=torch.bincount(torch.tensor(ME), minlength=len(EC)).float(),
                mut_x=torch.tensor(np.array(MF, float), dtype=torch.float32), mut_edge=torch.tensor(ME),
                mut_pos=torch.tensor(MP), mut_from=torch.tensor(MFr), mut_to=torch.tensor(MT),
                cand=torch.tensor(cand), log_x=torch.tensor(np.log(v.freqs), dtype=torch.float32))
