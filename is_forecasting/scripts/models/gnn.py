"""Basic edge-conditioned message-passing GNN on HA1+HA2 mutation trees.

  mutation encoder:  m = MLP(bio feats) and/or  E_pos[subtype, pos] + E_from[aa] + E_to[aa]
  edge embedding:    e = sum_{mutations on edge} m  + w * log1p(#mutations)
  node encoder:      h0 = MLP(node feats)
  K layers:          h <- LayerNorm(h + MLP([h, msg_from_parent, sum msg_from_children]))
                     msg_down = MLP_d([h_parent, e]),  msg_up = MLP_u([h_child, e])
  readout:           f = MLP(h) (last layer zero-init -> starts at naive)
  forecast:          p = softmax(log x + f) over season-t genotypes (Huddleston 2020, Eq. 1)
"""
import torch
import torch.nn as nn
from .graph_data import NODE_FEATS, MUT_FEATS

N_POS = {"h3n2": 550, "h1n1pdm": 549}
POS_OFFSET = {"h3n2": 0, "h1n1pdm": 550}


def mlp(i, h, o, drop=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


def batch_graphs(graphs, stats):
    """Concatenate graphs into one disjoint batch; standardize features with train stats."""
    xs, ec, ep, mx, me, mp, mf, mt, nm, cand, segs, logx = [], [], [], [], [], [], [], [], [], [], [], []
    no = eo = 0
    for gi, g in enumerate(graphs):
        xs.append(g["x"]); ec.append(g["edge_child"] + no); ep.append(g["edge_parent"] + no)
        mx.append(g["mut_x"]); me.append(g["mut_edge"] + eo); mp.append(g["mut_pos"] + POS_OFFSET[g["subtype"]])
        mf.append(g["mut_from"]); mt.append(g["mut_to"]); nm.append(g["n_mut_per_edge"])
        cand.append(g["cand"] + no); segs.append(torch.full((len(g["cand"]),), gi)); logx.append(g["log_x"])
        no += g["n_nodes"]; eo += len(g["edge_child"])
    b = dict(x=torch.cat(xs), edge_child=torch.cat(ec), edge_parent=torch.cat(ep), mut_x=torch.cat(mx),
             mut_edge=torch.cat(me), mut_pos=torch.cat(mp), mut_from=torch.cat(mf), mut_to=torch.cat(mt),
             n_mut=torch.cat(nm), cand=torch.cat(cand), seg=torch.cat(segs), log_x=torch.cat(logx),
             n_nodes=no, n_edges=eo, n_graphs=len(graphs))
    if all("seq" in g for g in graphs):  # global (position, residue) ids, subtype-specific position tables
        b["seq_ids"] = torch.cat([(g["seq"] + 21 * (POS_OFFSET[g["subtype"]] + torch.arange(g["seq"].shape[1]))[None, :]) for g in graphs]) \
            if len({g["seq"].shape[1] for g in graphs}) == 1 else None
        if b["seq_ids"] is None:  # pad to common length with a dedicated null id
            Lmax = max(g["seq"].shape[1] for g in graphs); null = 21 * sum(N_POS.values())
            b["seq_ids"] = torch.cat([torch.nn.functional.pad(g["seq"] + 21 * (POS_OFFSET[g["subtype"]] + torch.arange(g["seq"].shape[1]))[None, :],
                                                              (0, Lmax - g["seq"].shape[1]), value=null) for g in graphs])
    b["x"] = (b["x"] - stats["x_mu"]) / stats["x_sd"]
    b["mut_x"] = (b["mut_x"] - stats["m_mu"]) / stats["m_sd"]
    return b


def feature_stats(graphs):
    x = torch.cat([g["x"] for g in graphs]); m = torch.cat([g["mut_x"] for g in graphs])
    return dict(x_mu=x.mean(0), x_sd=x.std(0).clamp_min(1e-3), m_mu=m.mean(0), m_sd=m.std(0).clamp_min(1e-3))


class MutationTreeGNN(nn.Module):
    def __init__(self, mut_repr="bio", hidden=32, layers=4, dropout=0.1, use_seq=False, n_node_feats=None):
        super().__init__()
        self.mut_repr, self.layers, self.use_seq = mut_repr, layers, use_seq
        if use_seq:  # node sequence: sum of (position, residue) embeddings + direct linear fitness term (nests the linear model)
            n_ids = 21 * sum(N_POS.values()) + 1  # +1 padding id
            self.seq_emb = nn.EmbeddingBag(n_ids, hidden, mode="sum", padding_idx=n_ids - 1)
            nn.init.normal_(self.seq_emb.weight, std=0.01)
            self.seq_beta = nn.EmbeddingBag(n_ids, 1, mode="sum", padding_idx=n_ids - 1)
            nn.init.zeros_(self.seq_beta.weight)
        d = hidden
        if mut_repr in ("bio", "both"):
            self.mut_mlp = mlp(len(MUT_FEATS), d, d, dropout)
        if mut_repr in ("emb", "both"):
            self.pos_emb = nn.Embedding(sum(N_POS.values()), d); self.from_emb = nn.Embedding(21, d); self.to_emb = nn.Embedding(21, d)
            for e in (self.pos_emb, self.from_emb, self.to_emb): nn.init.normal_(e.weight, std=0.1)
        self.nmut = nn.Linear(1, d)
        self.node_enc = mlp(n_node_feats or len(NODE_FEATS), d, d, dropout)
        self.down = nn.ModuleList([mlp(2 * d, d, d, dropout) for _ in range(layers)])
        self.up = nn.ModuleList([mlp(2 * d, d, d, dropout) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(3 * d, d, d, dropout) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.readout = mlp(d, d, 1, dropout)
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)

    def edge_embedding(self, b):
        m = 0
        if self.mut_repr in ("bio", "both"): m = m + self.mut_mlp(b["mut_x"])
        if self.mut_repr in ("emb", "both"): m = m + self.pos_emb(b["mut_pos"]) + self.from_emb(b["mut_from"]) + self.to_emb(b["mut_to"])
        e = torch.zeros(b["n_edges"], m.shape[1]).index_add_(0, b["mut_edge"], m)
        return e + self.nmut(torch.log1p(b["n_mut"])[:, None])

    def fitness(self, b):
        h = self.node_enc(b["x"])
        if self.use_seq:
            h = h + self.seq_emb(b["seq_ids"])
        if self.layers:
            e = self.edge_embedding(b)
        for k in range(self.layers):
            hc, hp = h[b["edge_child"]], h[b["edge_parent"]]
            m_down = torch.zeros_like(h).index_add_(0, b["edge_child"], self.down[k](torch.cat([hp, e], 1)))
            m_up = torch.zeros_like(h).index_add_(0, b["edge_parent"], self.up[k](torch.cat([hc, e], 1)))
            h = self.norm[k](h + self.upd[k](torch.cat([h, m_down, m_up], 1)))
        f = self.readout(h[b["cand"]]).squeeze(1)
        if self.use_seq:
            f = f + self.seq_beta(b["seq_ids"][b["cand"]]).squeeze(1)
        return f

    def log_probs(self, b, f=None):
        """Per-graph log-softmax(log x + f) over candidates; returns flat tensor aligned with b['cand']."""
        z = b["log_x"] + (self.fitness(b) if f is None else f)
        out = torch.empty_like(z)
        for gi in range(b["n_graphs"]):
            idx = (b["seg"] == gi).nonzero().squeeze(1)
            out[idx] = torch.log_softmax(z[idx], 0)
        return out
