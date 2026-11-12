"""Local branching index (Neher et al. 2014) on a Nextstrain time tree, following augur's algorithm.

Branch lengths in years (num_date differences). A tip is 'alive' if its date is in (cutoff - window, cutoff]; an internal node is alive if any retained child
is; only branches above alive nodes contribute new branch length (augur.lbi). Normalized by the max over nodes.
Can be restricted to the subtree spanned by leaves dated <= cutoff (pruning), for leak-free history.
"""
import numpy as np


def compute_lbi(root, cutoff, tau=0.3, window=0.5, keep=None):
    """root: auspice_io.Node. cutoff: decimal year. keep: set of tip names to retain (None = all).
    Returns {node_name: lbi} over retained nodes (a node is retained if it has a retained descendant tip)."""
    order, parent = [], {}
    stack = [root]
    while stack:  # preorder
        n = stack.pop(); order.append(n)
        for c in n.children:
            parent[id(c)] = n; stack.append(c)
    kept = {}
    for n in reversed(order):  # postorder: retention
        kept[id(n)] = (n.name in keep) if (n.is_tip and keep is not None) else (n.is_tip or any(kept[id(c)] for c in n.children))
    alive = {}
    for n in reversed(order):  # augur select_nodes_in_season: tips in (cutoff - window, cutoff]; internal = any child alive
        if n.is_tip:
            alive[id(n)] = kept[id(n)] and (cutoff - window < n.num_date <= cutoff)
        else:
            alive[id(n)] = any(alive[id(c)] for c in n.children if kept[id(c)])
    up, down = {}, {}
    for n in reversed(order):  # postorder
        if not kept[id(n)]: continue
        u = sum(up[id(c)] for c in n.children if kept[id(c)])
        p = parent.get(id(n))
        bl = max(n.num_date - p.num_date, 0.0) / tau if p is not None else 0.0
        u *= np.exp(-bl)
        if alive[id(n)]: u += tau * (1 - np.exp(-bl))
        up[id(n)] = u
    for n in order:  # preorder
        if not kept[id(n)]: continue
        p = parent.get(id(n))
        if p is None:
            down[id(n)] = 0.0; continue
        d = down[id(p)] + sum(up[id(c)] for c in p.children if c is not n and kept[id(c)])
        bl = max(n.num_date - p.num_date, 0.0) / tau
        d *= np.exp(-bl)
        if alive[id(n)]: d += tau * (1 - np.exp(-bl))  # augur uses the child's alive flag
        down[id(n)] = d
    lbi = {n.name: down[id(n)] + sum(up[id(c)] for c in n.children if kept[id(c)]) for n in order if kept[id(n)]}
    mx = max(lbi.values()) or 1.0
    return {k: v / mx for k, v in lbi.items()}
