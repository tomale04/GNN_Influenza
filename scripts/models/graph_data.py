"""Build GNN-ready graphs from HA1+HA2 mutation trees.

Uses only information in tree t (available at snapshot t). Windows relative to season t:
  W0 = [Oct 1 (t-1), snapshot]   (current season; == n_recent)
  W1 = [Apr 1 (t-1), Oct 1 (t-1))
  W2 = [Oct 1 (t-2), Apr 1 (t-1))
Node features are node-local (own leaves only); anything clade-level must come from message passing.
Each edge (parent -> child) carries the set of AA mutations on it, described by
  bio features: HA1?, epitope?, Koel site?, BLOSUM62, d(hydropathy), d(charge), d(volume), reversion?,
                homoplasy (other edges with same mutation), to-residue frequency at site in W0/W1/W2,
                from-residue frequency at site in W0/W1/W2
  ids for learned embeddings: global position index (subtype-specific), from-AA, to-AA
"""
import os, sys, json, datetime, functools, collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import forecast_framework as ff
from auspice_io import load_dataset, iter_preorder
from build_mutation_trees import tip_date
from Bio.Align import substitution_matrices

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_ID = {a: i + 1 for i, a in enumerate(AAS)}  # 0 = other / unknown
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
CHARGE = dict(D=-1, E=-1, K=1, R=1, H=0.1)
VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
BLOSUM = substitution_matrices.load("BLOSUM62")
SITES = json.load(open(os.path.join(ff.ROOT, "data", "annotations", "sites.json")))
TREES_DIR = os.path.join(ff.ROOT, "data", "nextstrain_trees")
CACHE = os.path.join(ff.ROOT, "data", "graph_cache")

NODE_FEATS = ["freq_W0", "freq_W1", "freq_W2", "logc_W0", "logc_W1", "logc_W2", "has_leaves",
              "yrs_since_first", "yrs_since_last", "rel_depth", "is_candidate", "is_root"]
MUT_FEATS = ["is_HA1", "epitope", "koel", "blosum", "d_hydro", "d_charge", "d_volume", "reversion",
             "log_homoplasy", "to_W0", "to_W1", "to_W2", "from_W0", "from_W1", "from_W2"]


def _yrs(d0, d1):
    return (d1 - d0).days / 365.25


@functools.lru_cache(maxsize=None)
def build_graph(subtype, year):
    path = os.path.join(CACHE, f"{subtype}_{year}.pt")
    if os.path.exists(path):
        return torch.load(path)
    mt = ff.load_tree(subtype, year)
    nodes = mt["nodes"]
    snap = datetime.date.fromisoformat(mt["meta"]["snapshot_date"])
    w = [(datetime.date(year - 1, 10, 1), snap + datetime.timedelta(days=1)),
         (datetime.date(year - 1, 4, 1), datetime.date(year - 1, 10, 1)),
         (datetime.date(year - 2, 10, 1), datetime.date(year - 1, 4, 1))]
    ds = load_dataset(os.path.join(TREES_DIR, subtype, str(year)))
    date = {n.name: tip_date(n) for n in iter_preorder(ds["tree"]) if n.is_tip}

    N = len(nodes); L1 = len(nodes[0]["HA1"])
    seqs = [n["HA1"] + n["HA2"] for n in nodes]
    cnt = np.zeros((N, 3)); first = [None] * N; last = [None] * N
    for i, n in enumerate(nodes):
        ds_ = [date[l] for l in n["leaves"]]
        for k, (a, b) in enumerate(w):
            cnt[i, k] = sum(a <= d < b for d in ds_)
        if ds_: first[i], last[i] = min(ds_), max(ds_)
    assert (cnt[:, 0] == np.array([n["n_recent"] for n in nodes])).all()
    tot = cnt.sum(0); tot[tot == 0] = 1
    freq = cnt / tot

    # residue frequencies per site and window: res[k][pos][aa]
    res = [collections.defaultdict(collections.Counter) for _ in range(3)]
    for i in range(N):
        for k in range(3):
            if cnt[i, k]:
                for p, a in enumerate(seqs[i]): res[k][p][a] += freq[i, k]

    depth = np.array([len(n["mutations_from_reference"]) for n in nodes], float)
    cand = cnt[:, 0] > 0
    rel_depth = depth - (depth[cand] * freq[cand, 0]).sum()
    X = np.zeros((N, len(NODE_FEATS)))
    X[:, 0:3] = freq; X[:, 3:6] = np.log1p(cnt)
    for i in range(N):
        if first[i]:
            X[i, 6] = 1; X[i, 7] = _yrs(first[i], snap); X[i, 8] = _yrs(last[i], snap)
    X[:, 9] = rel_depth / 10; X[:, 10] = cand; X[0, 11] = 1

    # mutations on edges
    epi = set(SITES[subtype]["epitope"]); koel = set(SITES[subtype]["koel"])
    homo = collections.Counter(m for n in nodes for m in n["mutations"])
    ancestors_states = {0: collections.defaultdict(set)}
    for p, a in enumerate(seqs[0]): ancestors_states[0][p].add(a)
    M_feat, M_edge, M_pos, M_from, M_to, E_child, E_parent = [], [], [], [], [], [], []
    for i, n in enumerate(nodes):  # nodes are in preorder (parents before children)
        if n["parent"] is None: continue
        par = n["parent"]; e = len(E_child); E_child.append(i); E_parent.append(par)
        st = collections.defaultdict(set, {k: set(v) for k, v in ancestors_states[par].items()})
        for m in n["mutations"]:
            gene, rest = m.split(":"); f, pos, t = rest[0], int(rest[1:-1]), rest[-1]
            gp = (pos - 1) if gene == "HA1" else (L1 + pos - 1)
            rev = t in ancestors_states[par][gp]
            fv = [gene == "HA1", gene == "HA1" and pos in epi, gene == "HA1" and pos in koel,
                  BLOSUM[f][t] / 4 if f in AAS and t in AAS else 0,
                  (KD.get(t, 0) - KD.get(f, 0)) / 4.5, CHARGE.get(t, 0) - CHARGE.get(f, 0),
                  (VOLUME.get(t, 120) - VOLUME.get(f, 120)) / 100, rev, np.log1p(homo[m] - 1)]
            fv += [res[k][gp][t] for k in range(3)] + [res[k][gp][f] for k in range(3)]
            M_feat.append(fv); M_edge.append(e); M_pos.append(gp); M_from.append(AA_ID.get(f, 0)); M_to.append(AA_ID.get(t, 0))
            st[gp].add(t)
        ancestors_states[i] = st
    g = dict(subtype=subtype, year=year, n_nodes=N, x=torch.tensor(X, dtype=torch.float32),
             edge_child=torch.tensor(E_child), edge_parent=torch.tensor(E_parent),
             n_mut_per_edge=torch.bincount(torch.tensor(M_edge), minlength=len(E_child)).float(),
             mut_x=torch.tensor(np.array(M_feat, float), dtype=torch.float32), mut_edge=torch.tensor(M_edge),
             mut_pos=torch.tensor(M_pos), mut_from=torch.tensor(M_from), mut_to=torch.tensor(M_to),
             cand=torch.tensor(np.where(cand)[0]), log_x=torch.tensor(np.log(freq[cand, 0]), dtype=torch.float32))
    os.makedirs(CACHE, exist_ok=True); torch.save(g, path)
    return g


if __name__ == "__main__":
    for s in ff.SUBTYPES:
        for y in ff.YEARS:
            g = build_graph(s, y)
            print(s, y, "nodes", g["n_nodes"], "edges", len(g["edge_child"]), "muts", len(g["mut_edge"]), "cands", len(g["cand"]),
                  "| mean mut feats", np.round(g["mut_x"].mean(0).numpy(), 2).tolist())


@functools.lru_cache(maxsize=None)
def node_seq_ids(subtype, year):
    """[n_nodes, L] residue ids (0 = gap/X/other, 1..20 = AAS) for each node's HA1+HA2 sequence."""
    nodes = ff.load_tree(subtype, year)["nodes"]
    return torch.tensor([[AA_ID.get(a, 0) for a in n["HA1"] + n["HA2"]] for n in nodes])
