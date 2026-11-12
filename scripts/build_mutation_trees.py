"""Collapse Nextstrain HA trees into HA1+HA2 amino-acid mutation trees.

Each mutation-tree node is an HA1+HA2 protein genotype on the ancestry of the samples; each edge
carries >=1 HA1/HA2 AA mutation. Nextstrain edges with no (effective) HA1/HA2 AA mutations are
collapsed into their parent.

Rules:
- Only HA1 and HA2 amino-acid mutations are used (SigPep and synonymous changes are ignored).
- Mutations to '-' or 'X' (gaps / ambiguous) are ignored. A mutation from '-'/'X' to a residue is
  kept only if the residue differs from the tracked genotype state (the from-state is rewritten).
- Mutations on the Nextstrain root node (reference -> root in newer exports) define the root
  genotype rather than an edge.
- n_recent counts leaves mapping to the node with collection date in [Oct 1 of previous year,
  snapshot date] using Nextstrain's num_date point estimate.

Usage: python3 -I build_mutation_trees.py TREES_DIR OUT_DIR
"""
import sys, os, json, math, datetime, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auspice_io import load_dataset, validate, repair_root_nuc, GENES

REFERENCE_YEAR = 2026  # root_sequence of 2024+ builds is the build reference (root node carries ref->root muts)
SUBTYPES = ["h3n2", "h1n1pdm"]


def num_to_date(x):
    y = math.floor(x)
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    doy = min(max(round((x - y) * n + 0.5), 1), n)  # augur v2 exports: num = y + (doy - 0.5)/n (may be rounded to 3 dp)
    return datetime.date(y, 1, 1) + datetime.timedelta(days=doy - 1)


def tip_date(n):
    """v1 (2019) exports use num = y + doy/365.25 but also carry an explicit 'date' string; prefer it."""
    d = (n.attrs or {}).get("date")
    if isinstance(d, str) and len(d) == 10 and "X" not in d:
        return datetime.date.fromisoformat(d)
    return num_to_date(n.num_date)


def diff(seq, ref):
    return [f"{g}:{ref[g][i]}{i+1}{seq[g][i]}" for g in GENES for i in range(len(ref[g])) if seq[g][i] != ref[g][i]]


def build(ds, year, snapshot_date, ref):
    win_start = datetime.date(year - 1, 10, 1)
    root = ds["tree"]
    geno0 = {g: list(ds["root_seq"][g]) for g in GENES}
    nodes = []

    def new_node(parent_id, muts, geno, ns_node):
        nodes.append(dict(id=len(nodes), parent=parent_id, mutations=muts, geno=geno, nextstrain_nodes=[ns_node.name],
                          leaves=[], recent_leaves=[], children=[]))
        if parent_id is not None: nodes[parent_id]["children"].append(len(nodes) - 1)
        return len(nodes) - 1

    def effective(ns_node, geno):
        out = []
        for g in GENES:
            for f, p, t in ns_node.aa_muts[g]:
                if t in "-X": continue
                cur = geno[g][p - 1]
                if f not in "-X" and cur != f:
                    raise ValueError(f"from-state mismatch {g}:{f}{p}{t} at {ns_node.name} (have {cur})")
                if t != cur: out.append((g, cur, p, t))
        return out

    # root genotype absorbs root-node mutations
    g = {k: v[:] for k, v in geno0.items()}
    for gg, f, p, t in effective(root, g): g[gg][p - 1] = t
    rid = new_node(None, [], g, root)
    stats = collections.Counter()
    stack = [(c, rid) for c in reversed(root.children)]
    leaf_records = [(root, rid)] if root.is_tip else []
    while stack:
        n, pid = stack.pop()
        pg = nodes[pid]["geno"]
        muts = effective(n, pg)
        if muts:
            g = {k: v[:] for k, v in pg.items()}
            for gg, f, p, t in muts: g[gg][p - 1] = t
            mid = new_node(pid, [f"{gg}:{f}{p}{t}" for gg, f, p, t in muts], g, n)
        else:
            mid = pid; nodes[pid]["nextstrain_nodes"].append(n.name); stats["collapsed_edges"] += 1
        if n.is_tip: leaf_records.append((n, mid))
        stack.extend((c, mid) for c in reversed(n.children))

    for n, mid in leaf_records:
        d = tip_date(n)
        nodes[mid]["leaves"].append(n.name)
        if win_start <= d <= snapshot_date:
            nodes[mid]["recent_leaves"].append(n.name)
            conf = n.date_conf or [n.num_date, n.num_date]
            if conf[1] - conf[0] > 2 / 365: stats["recent_leaves_with_ambiguous_date"] += 1
        elif d > snapshot_date:
            stats["leaves_after_snapshot"] += 1

    seen = collections.Counter()
    out_nodes = []
    for nd in nodes:
        seq = {g: "".join(nd["geno"][g]) for g in GENES}
        seen[seq["HA1"] + seq["HA2"]] += 1
        out_nodes.append(dict(id=nd["id"], parent=nd["parent"], children=nd["children"], mutations=nd["mutations"],
                              HA1=seq["HA1"], HA2=seq["HA2"], mutations_from_reference=diff(seq, ref),
                              n_leaves=len(nd["leaves"]), n_recent=len(nd["recent_leaves"]),
                              recent_leaves=nd["recent_leaves"], leaves=nd["leaves"], nextstrain_nodes=nd["nextstrain_nodes"]))
    stats.update(n_nodes=len(out_nodes), n_leaves=len(leaf_records), n_recent=sum(n["n_recent"] for n in out_nodes),
                 nodes_with_recent=sum(n["n_recent"] > 0 for n in out_nodes),
                 genotypes_repeated_in_tree=sum(v - 1 for v in seen.values() if v > 1))
    assert all(len(n["mutations"]) >= 1 for n in out_nodes[1:])
    return out_nodes, dict(stats), win_start


def main(trees_dir, out_dir):
    manifest = {(r["subtype"], r["year"]): r for r in json.load(open(os.path.join(trees_dir, "manifest.json")))}
    os.makedirs(os.path.join(out_dir, "references"), exist_ok=True)
    summary = []
    for s in SUBTYPES:
        rds = load_dataset(os.path.join(trees_dir, s, str(REFERENCE_YEAR)))
        ref = {g: rds["root_seq"][g] for g in GENES}
        with open(os.path.join(out_dir, "references", f"{s}_HA1_HA2.fasta"), "w") as fh:
            for g in GENES: fh.write(f">{s}_{g} nextstrain seasonal-flu {s} HA reference (root_sequence of {REFERENCE_YEAR} build)\n{ref[g]}\n")
        for y in sorted(yy for ss, yy in manifest if ss == s):
            ds = load_dataset(os.path.join(trees_dir, s, str(y)))
            st = validate(ds)
            fixes = []
            if st["nuc_mismatch"] or st["aa_mismatch"]:
                fixes = repair_root_nuc(ds); st = validate(ds)
                assert st["nuc_mismatch"] == 0 and st["aa_mismatch"] == 0, (s, y, st)
            assert all(len(ds["root_seq"][g]) == len(ref[g]) for g in GENES), "AA coordinate system differs from reference"
            main_file = next(f for f in manifest[(s, y)]["files"] if f["key"].endswith(("ha_12y.json", "ha_12y_tree.json")))
            snap = datetime.date.fromisoformat(main_file["last_modified"][:10])
            nodes, stats, win_start = build(ds, y, snap, ref)
            meta = dict(subtype=s, year=y, snapshot_date=str(snap), window=[str(win_start), str(snap)],
                        source=dict(file=main_file["key"], version_id=main_file["version_id"], s3_last_modified=main_file["last_modified"],
                                    auspice_version=ds["version"], root_sequence_source=ds["root_source"],
                                    root_sequence_repairs=[f"nuc {p}: {a}->{b}" for p, a, b in fixes]),
                        reference=f"references/{s}_HA1_HA2.fasta", coordinates="HA1/HA2 amino-acid positions, 1-based (same for all years)",
                        validation={k: v for k, v in st.items() if k != "mismatch_examples"}, stats=stats)
            d = os.path.join(out_dir, s); os.makedirs(d, exist_ok=True)
            json.dump(dict(meta=meta, nodes=nodes), open(os.path.join(d, f"{y}.json"), "w"))
            summary.append(dict(subtype=s, year=y, snapshot=str(snap), repaired=len(fixes), **stats))
            print(s, y, stats, "root repairs:", len(fixes))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
