"""Unified loading of Nextstrain auspice v1/v2 seasonal-flu HA JSONs + root sequences.

Genotype convention: sequence(node) = root_sequence with every mutation on the path from the
root applied, *including mutations on the root node itself* (newer augur exports put
reference->root mutations on the root node; older exports have an empty root branch and the
root_sequence is the inferred root).
"""
import json, os, glob

GENES = ("HA1", "HA2")
CODON = {}
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for i, a in enumerate(_b):
    for j, b in enumerate(_b):
        for k, c in enumerate(_b):
            CODON[a + b + c] = _aa[16 * i + 4 * j + k]


def translate(nt):
    return "".join(CODON.get(nt[i:i + 3].upper(), "X") for i in range(0, len(nt) - len(nt) % 3, 3))


class Node:
    __slots__ = ("name", "children", "parent", "nuc_muts", "aa_muts", "is_tip", "num_date", "date_conf", "attrs")

    def __init__(self):
        self.children, self.parent = [], None


def _parse_mut(m):
    return m[0], int(m[1:-1]), m[-1]


def load_dataset(dirpath):
    """Return dict(tree=root Node, root_seq={nuc,HA1,HA2,...}, annotations={gene:(start0,end0)}, meta, version, root_source)."""
    main = [f for f in glob.glob(os.path.join(dirpath, "*ha_12y.json")) + glob.glob(os.path.join(dirpath, "*ha_12y_tree.json"))]
    assert len(main) == 1, main
    j = json.load(open(main[0]))
    v2 = "tree" in j
    if v2:
        meta = j["meta"]
        ann = {g: (a["start"] - 1, a["end"]) for g, a in meta["genome_annotations"].items()}  # 1-based incl -> 0-based half-open
        troot = j["tree"]
    else:
        meta = json.load(open(main[0].replace("_tree.json", "_meta.json")))
        ann = {g: (a["start"], a["end"]) for g, a in meta["annotations"].items()}  # already 0-based half-open
        troot = j

    # root sequence: embedded > sidecar file
    if j.get("root_sequence"):
        root_seq, root_source = dict(j["root_sequence"]), "embedded"
    else:
        rs = glob.glob(os.path.join(dirpath, "*root-sequence.json"))
        root_seq, root_source = (json.load(open(rs[0])), os.path.basename(rs[0])) if rs else (None, None)
    if root_seq is not None:
        for g in ("SigPep",) + GENES:
            s0, e0 = ann[g]
            tr = translate(root_seq["nuc"][s0:e0])
            if g in root_seq and root_seq[g].rstrip("*") != tr.rstrip("*"):
                raise ValueError(f"{dirpath}: provided {g} root differs from translation of nuc root")
            root_seq[g] = tr.rstrip("*")

    def build(d, parent):
        n = Node()
        n.parent = parent
        if v2:
            n.name = d["name"]
            mm = d.get("branch_attrs", {}).get("mutations", {}) or {}
            nuc = mm.get("nuc", [])
            aa = {g: mm.get(g, []) for g in GENES}
            a = d.get("node_attrs", {})
            nd = a.get("num_date", {})
            n.num_date, n.date_conf = nd.get("value"), nd.get("confidence")
        else:
            n.name = d["strain"]
            nuc = d.get("muts", []) or []
            aa = {g: (d.get("aa_muts", {}) or {}).get(g, []) for g in GENES}
            a = d.get("attr", {})
            n.num_date, n.date_conf = a.get("num_date"), a.get("num_date_confidence")
        n.nuc_muts = [_parse_mut(m) for m in nuc]
        n.aa_muts = {g: [_parse_mut(m) for m in aa[g]] for g in GENES}
        n.attrs = a
        n.children = [build(c, n) for c in d.get("children", [])]
        n.is_tip = not n.children
        return n

    return dict(tree=build(troot, None), root_seq=root_seq, annotations=ann, meta=meta,
                version="v2" if v2 else "v1", root_source=root_source, path=main[0])


def iter_preorder(n):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(reversed(x.children))


def validate(ds):
    """Walk tree applying mutations; count from-state mismatches for nuc and HA1/HA2 AA,
    and check AA translation of reconstructed nuc matches AA reconstruction at tips."""
    root_seq, ann = ds["root_seq"], ds["annotations"]
    stats = {"nuc_mut": 0, "nuc_mismatch": 0, "aa_mut": 0, "aa_mismatch": 0, "tips_checked": 0, "tips_translation_mismatch": 0, "mismatch_examples": []}
    stack = [(ds["tree"], list(root_seq["nuc"]), {g: list(root_seq[g]) for g in GENES})]
    while stack:
        n, nuc, aa = stack.pop()
        nuc = nuc[:]; aa = {g: v[:] for g, v in aa.items()}
        for f, p, t in n.nuc_muts:
            stats["nuc_mut"] += 1
            if nuc[p - 1] != f and f not in "-N":
                stats["nuc_mismatch"] += 1
                if len(stats["mismatch_examples"]) < 5: stats["mismatch_examples"].append((n.name, f"nuc {f}{p}{t} have {nuc[p-1]}"))
            nuc[p - 1] = t
        for g in GENES:
            for f, p, t in n.aa_muts[g]:
                stats["aa_mut"] += 1
                if aa[g][p - 1] != f and f not in "-X":
                    stats["aa_mismatch"] += 1
                    if len(stats["mismatch_examples"]) < 10: stats["mismatch_examples"].append((n.name, f"{g} {f}{p}{t} have {aa[g][p-1]}"))
                aa[g][p - 1] = t
        if n.is_tip:
            stats["tips_checked"] += 1
            s = "".join(nuc)
            for g in GENES:
                tr = translate(s[ann[g][0]:ann[g][1]]).rstrip("*")
                # compare only where both are unambiguous residues
                if any(x != y and x not in "X-" and y not in "X-" for x, y in zip(tr, aa[g])):
                    stats["tips_translation_mismatch"] += 1; break
        for c in n.children:
            stack.append((c, nuc, aa))
    return stats


def repair_root_nuc(ds):
    """Fix root nuc sequence using from-states of the first mutation at each position on every
    root-to-tip path (these must equal the root state). Raises on conflicting evidence.
    Re-derives HA1/HA2/SigPep root AA by translation. Returns list of (pos, old, new)."""
    cons = {}
    stack = [(ds["tree"], frozenset())]
    while stack:
        n, seen = stack.pop()
        new = set(seen)
        for f, p, t in n.nuc_muts:
            if p not in seen and f not in "-N":
                if cons.setdefault(p, f) != f:
                    raise ValueError(f"conflicting root evidence at nuc {p}")
            new.add(p)
        new = frozenset(new)
        stack.extend((c, new) for c in n.children)
    nuc = list(ds["root_seq"]["nuc"]); fixes = []
    for p, f in sorted(cons.items()):
        if nuc[p - 1] != f:
            fixes.append((p, nuc[p - 1], f)); nuc[p - 1] = f
    ds["root_seq"]["nuc"] = "".join(nuc)
    for g in ("SigPep",) + GENES:
        s0, e0 = ds["annotations"][g]
        ds["root_seq"][g] = translate(ds["root_seq"]["nuc"][s0:e0]).rstrip("*")
    return fixes
