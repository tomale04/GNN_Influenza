# File for processing and storing the nextstrain data

"""
nextstrain_preprocess.py
=========================
 
Flag nodes in a Nextstrain phylogeny that look low quality or may have data 
problems. Note that this script will not make any changes to the actual data.
It just reads the raw JSON files and writes a table with one row per tree node 
and  one boolean column per check, so that we can decide what to do with the 
flagged late on.
 
Inputs
------------------
--tree          Main JSON                (e.g. seasonal-flu_h1n1pdm_ha_12y.json)
--tip-freq      Tip-frequency JSON       (e.g. seasonal-flu_h1n1pdm_ha_12y_tip-frequencies.json)
 
Outputs (written to --output-dir)
---------------------------------
<subtype>_node_flags.csv        One row per node, with flag columns + reasons
<subtype>_attribute_audit.csv   Per-attribute missing/placeholder counts (tips vs internal)
<subtype>_flag_report.txt       Human-readable summary (also printed to the console)
 
Execution copy-paste for h1n1 and h3n2:
-------
python data/nextstrain_preprocess.py \
    --tree data/raw/h1n1pdm/seasonal-flu_h1n1pdm_ha_12y.json \
    --tip-freq data/raw/h1n1pdm/seasonal-flu_h1n1pdm_ha_12y_tip-frequencies.json \
    --subtype h1n1pdm \
    --output-dir data/flagged/h1n1pdm

python data/nextstrain_preprocess.py \
    --tree data/raw/h3n2/seasonal-flu_h3n2_ha_12y.json \
    --tip-freq data/raw/h3n2/seasonal-flu_h3n2_ha_12y_tip-frequencies.json \
    --subtype h3n2 \
    --output-dir data/flagged/h3n2
"""
 
import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
 
# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #
 
# Strings treated as NA.
PLACEHOLDERS = {"", "unassigned", "unknown", "na", "n/a", "none", "?", "undefined"}
 
TIP_CATEGORICAL_ATTRS = ["region", "country", "division", "clade_membership", "subclade"]
NUMERIC_ATTRS = ["div", "num_date", "lbi", "ha_muts", "ha1_muts", "ep", "glyc"]
NON_NEGATIVE_ATTRS = ["div", "lbi", "ha_muts", "ha1_muts", "ep", "glyc"]
 
# Ambiguities in amino acid or nucleotides.
AMBIGUOUS_AA = set("X-*")
AMBIGUOUS_NUC = set("NRYKMSWBDHV-")

FLAG_DESCRIPTIONS = {
    "flag_duplicate_name": "Node name appears more than once in the tree.",
    "flag_qc_bad": "Tip has Nextclade QC overall status 'bad'.",
    "flag_qc_mediocre": "Tip has Nextclade QC overall status 'mediocre'.",
    "flag_qc_missing": "Tip QC status is missing or not one of good/mediocre/bad.",
    "flag_missing_attribute": "A required attribute is absent (see 'notes' column).",
    "flag_placeholder_value": "A tip categorical attribute holds a placeholder such as 'unassigned'/'unknown'.",
    "flag_invalid_numeric": "A numeric attribute is present but not a finite number.",
    "flag_negative_value": "A non-negative attribute (div, lbi, ha_muts, ...) is negative.",
    "flag_ha1_gt_ha_muts": "ha1_muts is greater than ha_muts (inconsistent).",
    "flag_date_out_of_range": "Tip date lies outside the frequency pivot range (+/- margin).",
    "flag_imprecise_date": "Tip date has a wide confidence interval (likely an imprecise/imputed date).",
    "flag_date_before_parent": "Node is dated earlier than its parent (beyond tolerance).",
    "flag_div_decreasing": "Divergence decreases from parent to child (negative branch length).",
    "flag_long_branch": "Branch length is an outlier (Tukey fence on positive branch lengths).",
    "flag_ambiguous_mutation": "Branch carries a mutation to/from an ambiguity code, gap, or stop codon.",
    "flag_region_country_conflict": "Tip region disagrees with the majority region of its country.",
    "flag_no_frequency_entry": "Tip is in the tree but has no entry in the tip-frequency file.",
    "flag_bad_frequency": "Tip frequencies have wrong length, NaN/inf, or values outside [0, 1].",
}
FLAG_NAMES = list(FLAG_DESCRIPTIONS.keys())
 
 
# --------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# --------------------------------------------------------------------------- #
 
def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
 
 
def is_placeholder(x):
    return isinstance(x, str) and x.strip().lower() in PLACEHOLDERS
 
 
def attr_value(node_attrs, key):
    entry = node_attrs.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("value")
    return entry
 
 
def date_ci(node_attrs):
    """Return (lo, hi) of the num_date confidence interval."""
    entry = node_attrs.get("num_date")
    if isinstance(entry, dict):
        conf = entry.get("confidence")
        if isinstance(conf, (list, tuple)) and len(conf) == 2 and all(is_number(c) for c in conf):
            return conf[0], conf[1]
    return None, None
 
 
def load_json(path):
    """Load a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: {path} is not valid JSON ({e})")
 
 
# --------------------------------------------------------------------------- #
# CREATE A LIST OF RECORDS.
# --------------------------------------------------------------------------- #
 
def flatten_tree(tree_json, qc_key):
    """Walk the tree and return a list of plain dictionaries (records), 
    one per node. The record holds all the metadata for that node, and
    a link to its parent.
    """
    root = tree_json["tree"] if "tree" in tree_json else tree_json
    records = []

    stack = [(root, None, "root", 0)]
 
    while stack:
        node, parent_idx, node_id, depth = stack.pop()
        node_attrs = node.get("node_attrs", {}) or {}
        branch_attrs = node.get("branch_attrs", {}) or {}
        children = node.get("children", []) or []
 
        # Count mutations on the branch leading into this node.
        muts = branch_attrs.get("mutations", {}) or {}
        n_nuc = len(muts.get("nuc", []) or [])
        n_aa = sum(len(v or []) for gene, v in muts.items() if gene != "nuc")
 
        lo, hi = date_ci(node_attrs)
        idx = len(records)
        records.append({
            "idx": idx,
            "node_id": node_id,
            "name": node.get("name"),
            "parent_idx": parent_idx,
            "parent_id": records[parent_idx]["node_id"] if parent_idx is not None else None,
            "depth": depth,
            "is_tip": len(children) == 0,
            "div": attr_value(node_attrs, "div"),
            "num_date": attr_value(node_attrs, "num_date"),
            "date_ci_width": (hi - lo) if lo is not None else None,
            "lbi": attr_value(node_attrs, "lbi"),
            "ha_muts": attr_value(node_attrs, "ha_muts"),
            "ha1_muts": attr_value(node_attrs, "ha1_muts"),
            "ep": attr_value(node_attrs, "ep"),
            "glyc": attr_value(node_attrs, "glyc"),
            "region": attr_value(node_attrs, "region"),
            "country": attr_value(node_attrs, "country"),
            "division": attr_value(node_attrs, "division"),
            "clade_membership": attr_value(node_attrs, "clade_membership"),
            "subclade": attr_value(node_attrs, "subclade"),
            "qc_status": attr_value(node_attrs, qc_key),
            "n_aa_mutations": n_aa,
            "n_nuc_mutations": n_nuc,
            "branch_length": None,       # filled in below
            "freq_max": None,            # filled in by frequency check
            "freq_sum": None,            # filled in by frequency check
            "flags": {name: False for name in FLAG_NAMES},
            "notes": [],
            "_node_attrs": node_attrs,
            "_mutations": muts,
        })
        for i in range(len(children) - 1, -1, -1):
            child_id = f"{node_id}/{i}"
            stack.append((children[i], idx, child_id, depth + 1))
 
    # Compute branch lengths.
    for r in records:
        if r["parent_idx"] is not None:
            p = records[r["parent_idx"]]
            if is_number(r["div"]) and is_number(p["div"]):
                r["branch_length"] = r["div"] - p["div"]
    return records
 
 
# --------------------------------------------------------------------------- #
# SIMPLE CHECKS FOR ATTRIBUTES/VARIABLES.
# --------------------------------------------------------------------------- #
 
def check_duplicate_names(records):
    counts = Counter(r["name"] for r in records)
    for r in records:
        if counts[r["name"]] > 1:
            r["flags"]["flag_duplicate_name"] = True
            r["notes"].append(f"name appears {counts[r['name']]}x")
 
 
def check_qc_status(records):
    """Flag tips by Nextclade QC (Quality Control) status.."""
    for r in records:
        if not r["is_tip"]:
            continue
        status = r["qc_status"]
        s = status.strip().lower() if isinstance(status, str) else None
        if s == "bad":
            r["flags"]["flag_qc_bad"] = True
        elif s == "mediocre":
            r["flags"]["flag_qc_mediocre"] = True
        elif s != "good":
            r["flags"]["flag_qc_missing"] = True
            r["notes"].append(f"qc status={status!r}")
 
 
def check_missing_and_placeholders(records):
    """Missing required attributes (absent) vs placeholder values ('unassigned', ...)."""
    for r in records:
        for attr in ("div", "num_date"):
            if r[attr] is None:
                r["flags"]["flag_missing_attribute"] = True
                r["notes"].append(f"missing:{attr}")
        if not r["is_tip"]:
            continue
        if r["lbi"] is None:
            r["flags"]["flag_missing_attribute"] = True
            r["notes"].append("missing:lbi")
        for attr in TIP_CATEGORICAL_ATTRS:
            v = r[attr]
            if v is None:
                r["flags"]["flag_missing_attribute"] = True
                r["notes"].append(f"missing:{attr}")
            elif is_placeholder(v):
                r["flags"]["flag_placeholder_value"] = True
                r["notes"].append(f"placeholder:{attr}={v!r}")
 
 
def check_valid_numeric(records):
    for r in records:
        for attr in NUMERIC_ATTRS:
            v = r[attr]
            if v is None:
                continue
            if not is_number(v):
                r["flags"]["flag_invalid_numeric"] = True
                r["notes"].append(f"invalid_numeric:{attr}={v!r}")
            elif attr in NON_NEGATIVE_ATTRS and v < 0:
                r["flags"]["flag_negative_value"] = True
                r["notes"].append(f"negative:{attr}={v}")
        if is_number(r["ha1_muts"]) and is_number(r["ha_muts"]) and r["ha1_muts"] > r["ha_muts"]:
            r["flags"]["flag_ha1_gt_ha_muts"] = True
            r["notes"].append(f"ha1_muts({r['ha1_muts']})>ha_muts({r['ha_muts']})")
 
 
def check_dates(records, pivots, date_margin, max_ci_width, parent_tol):
    """Date range, imprecise tip dates, and parent/child date consistency."""
    pmin = min(pivots) if pivots else None
    pmax = max(pivots) if pivots else None
    for r in records:
        d = r["num_date"]
        if not is_number(d):
            continue
        if r["is_tip"]:
            if pmin is not None and (d < pmin - date_margin or d > pmax + date_margin):
                r["flags"]["flag_date_out_of_range"] = True
                r["notes"].append(f"date {d:.3f} outside pivots [{pmin:.3f}, {pmax:.3f}]")
            w = r["date_ci_width"]
            if w is not None and w > max_ci_width:
                r["flags"]["flag_imprecise_date"] = True
                r["notes"].append(f"date CI width={w:.3f}y")
        if r["parent_idx"] is not None:
            pd_ = records[r["parent_idx"]]["num_date"]
            if is_number(pd_) and d < pd_ - parent_tol:
                r["flags"]["flag_date_before_parent"] = True
                r["notes"].append(f"date {d:.3f} < parent {pd_:.3f}")
 
 
def check_divergence(records, iqr_mult):
    """Negative branch lengths and long-branch outliers."""
    for r in records:
        bl = r["branch_length"]
        if bl is not None and bl < -1e-9:
            r["flags"]["flag_div_decreasing"] = True
            r["notes"].append(f"branch_length={bl:.6g}")
 
    positive = [r["branch_length"] for r in records
                if r["branch_length"] is not None and r["branch_length"] > 0]
    threshold = None
    if len(positive) >= 4:
        q1, _, q3 = statistics.quantiles(positive, n=4)
        threshold = q3 + iqr_mult * (q3 - q1)
        for r in records:
            bl = r["branch_length"]
            if bl is not None and bl > threshold:
                r["flags"]["flag_long_branch"] = True
                r["notes"].append(f"long branch {bl:.6g} > {threshold:.6g}")
    return threshold
 
 
def check_ambiguous_mutations(records):
    """Flag branches whose mutation list involves an ambiguity code, gap or stop codon.
    NOTE: Nextstrain often masks ambiguous sites instead of listing them, so zero hits
    does not prove sequences are free of ambiguity."""
    for r in records:
        for gene, mut_list in r["_mutations"].items():
            amb = AMBIGUOUS_NUC if gene == "nuc" else AMBIGUOUS_AA
            for m in (mut_list or []):
                if isinstance(m, str) and len(m) >= 3 and (m[0] in amb or m[-1] in amb):
                    r["flags"]["flag_ambiguous_mutation"] = True
                    r["notes"].append(f"ambiguous mutation {gene}:{m}")
                    break  # one note per gene is enough
 
 
def check_region_country_consistency(records):
    """For each country, find its majority region among tips; flag disagreeing tips."""
    country_regions = defaultdict(Counter)
    for r in records:
        if r["is_tip"] and isinstance(r["country"], str) and isinstance(r["region"], str) \
                and not is_placeholder(r["country"]) and not is_placeholder(r["region"]):
            country_regions[r["country"]][r["region"]] += 1
    majority = {c: cnt.most_common(1)[0][0] for c, cnt in country_regions.items()}
    conflicts = {c: dict(cnt) for c, cnt in country_regions.items() if len(cnt) > 1}
    for r in records:
        if r["is_tip"] and r["country"] in majority and r["region"] in country_regions[r["country"]]:
            if r["region"] != majority[r["country"]]:
                r["flags"]["flag_region_country_conflict"] = True
                r["notes"].append(f"region={r['region']} but {r['country']} is mostly {majority[r['country']]}")
    return conflicts
 
 
# --------------------------------------------------------------------------- #
# FREQUENCY FILE
# --------------------------------------------------------------------------- #
 
def extract_frequencies(tip_json):
    """Return (pivots, {tip_name: frequencies_list}). Entries without a
    'frequencies' list are ignored."""
    pivots = tip_json.get("pivots", [])
    freqs = {}
    for k, v in tip_json.items():
        if isinstance(v, dict) and isinstance(v.get("frequencies"), list):
            freqs[k] = v["frequencies"]
    return pivots, freqs
 
 
def check_frequencies(records, pivots, freqs, freq_tol):
    """Some quality checks for the frequency data of each tip. 
    Returns a dict of 'findings'.
    """
    findings = {}
 
    findings["pivots_numeric"] = bool(pivots) and all(is_number(p) for p in pivots)
    findings["pivots_increasing"] = findings["pivots_numeric"] and all(
        b > a for a, b in zip(pivots, pivots[1:]))
 
    tip_names = {r["name"] for r in records if r["is_tip"]}
    findings["in_tree_not_in_freq"] = sorted(tip_names - set(freqs))
    findings["in_freq_not_in_tree"] = sorted(set(freqs) - tip_names)
 
    # Per-tip checks
    for r in records:
        if not r["is_tip"]:
            continue
        f = freqs.get(r["name"])
        if f is None:
            r["flags"]["flag_no_frequency_entry"] = True
            continue
        problems = []
        if len(f) != len(pivots):
            problems.append(f"len={len(f)} vs {len(pivots)} pivots")
        bad_vals = [x for x in f if not is_number(x) or x < -freq_tol or x > 1 + freq_tol]
        if bad_vals:
            problems.append(f"{len(bad_vals)} invalid values (e.g. {bad_vals[0]!r})")
        if problems:
            r["flags"]["flag_bad_frequency"] = True
            r["notes"].append("freq: " + "; ".join(problems))
        good = [x for x in f if is_number(x)]
        if good:
            r["freq_max"] = max(good)
            r["freq_sum"] = sum(good)
 
    col_sums = []
    for j in range(len(pivots)):
        s = sum(f[j] for f in freqs.values() if j < len(f) and is_number(f[j]))
        col_sums.append(s)
    findings["col_sum_min"] = min(col_sums) if col_sums else None
    findings["col_sum_max"] = max(col_sums) if col_sums else None
    findings["col_sum_mean"] = statistics.mean(col_sums) if col_sums else None
    return findings
 
 
# --------------------------------------------------------------------------- #
# FLAG SOME ATTRIBUTES.
# --------------------------------------------------------------------------- #
 
def attribute_audit(records):
    """For every attribute key seen anywhere, count present / missing / placeholder
    separately for tips and internal nodes."""
    all_keys = set()
    for r in records:
        all_keys.update(r["_node_attrs"].keys())
 
    rows = []
    for key in sorted(all_keys):
        row = {"attribute": key}
        for label, is_tip in (("tip", True), ("internal", False)):
            group = [r for r in records if r["is_tip"] == is_tip]
            present = missing = placeholder = 0
            for r in group:
                v = attr_value(r["_node_attrs"], key)
                if key not in r["_node_attrs"] or v is None:
                    missing += 1
                elif is_placeholder(v):
                    placeholder += 1
                else:
                    present += 1
            row[f"{label}_n"] = len(group)
            row[f"{label}_present"] = present
            row[f"{label}_missing"] = missing
            row[f"{label}_placeholder"] = placeholder
        rows.append(row)
    return rows
 
 
# --------------------------------------------------------------------------- #
# WRITE OUTPUT.
# --------------------------------------------------------------------------- #
 
def finalize_flags(records):
    """Add summary columns: n_flags, any_flag, flag_reasons."""
    for r in records:
        active = [name for name in FLAG_NAMES if r["flags"][name]]
        r["n_flags"] = len(active)
        r["any_flag"] = bool(active)
        r["flag_reasons"] = ";".join(active)
 
 
def write_node_table(records, path):
    cols = ["node_id", "name", "parent_id", "parent_name", "is_tip", "depth", "div", 
            "branch_length", "num_date", "date_ci_width", "lbi", "ha_muts", "ha1_muts", 
            "ep", "glyc", "region", "country", "division", "clade_membership", "subclade",
            "qc_status", "n_aa_mutations", "n_nuc_mutations", "freq_max", "freq_sum"]
    header = cols + FLAG_NAMES + ["n_flags", "any_flag", "flag_reasons", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            parent_name = records[r["parent_idx"]]["name"] if r["parent_idx"] is not None else ""
            row = [parent_name if c == "parent_name" else r.get(c) for c in cols]
            row += [r["flags"][n] for n in FLAG_NAMES]
            row += [r["n_flags"], r["any_flag"], r["flag_reasons"], " | ".join(r["notes"])]
            w.writerow(["" if v is None else v for v in row])
 
 
def write_audit(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
 
 
def build_report(subtype, records, freq_findings, conflicts, long_threshold, args, qc_dist):
    tips = [r for r in records if r["is_tip"]]
    L = []
    L.append(f"FLAG REPORT: {subtype}")
    L.append("=" * 60)
    L.append(f"Nodes: {len(records)}   Tips: {len(tips)}   Internal: {len(records) - len(tips)}")
    L.append(f"Nodes with at least one flag: {sum(r['any_flag'] for r in records)}"
             f"  (tips: {sum(r['any_flag'] for r in tips)})")
    L.append("")
    L.append("Nextclade QC status among tips: " + ", ".join(f"{k}={v}" for k, v in qc_dist.most_common()))
    L.append("")
    L.append(f"{'flag':32s} {'tips':>8s} {'internal':>9s}   description")
    L.append("-" * 100)
    for name in FLAG_NAMES:
        t = sum(r["flags"][name] for r in tips)
        i = sum(r["flags"][name] for r in records if not r["is_tip"])
        L.append(f"{name:32s} {t:8d} {i:9d}   {FLAG_DESCRIPTIONS[name]}")
    L.append("")
    L.append("Frequency file checks")
    L.append("-" * 60)
    L.append(f"Pivots numeric: {freq_findings['pivots_numeric']}   strictly increasing: {freq_findings['pivots_increasing']}")
    L.append(f"Tree tips missing from frequency file: {len(freq_findings['in_tree_not_in_freq'])}")
    L.append(f"Frequency entries not matching any tree tip: {len(freq_findings['in_freq_not_in_tree'])}")
    if freq_findings["col_sum_mean"] is not None:
        L.append(f"Sum of frequencies across tips per pivot: min={freq_findings['col_sum_min']:.4f} "
                 f"mean={freq_findings['col_sum_mean']:.4f} max={freq_findings['col_sum_max']:.4f} (expected ~1)")
        if abs(freq_findings["col_sum_min"] - 1) > args.freq_sum_tol or abs(freq_findings["col_sum_max"] - 1) > args.freq_sum_tol:
            L.append("  Per-pivot totals deviate from 1 by more than the tolerance.")
    L.append("")
    L.append("More information")
    L.append("-" * 60)
    if long_threshold is not None:
        L.append(f"Long-branch threshold (Q3 + {args.long_branch_iqr}*IQR of positive branch lengths): {long_threshold:.6g}")
    else:
        L.append("Long-branch check skipped (too few positive branch lengths).")
    if conflicts:
        L.append(f"Countries assigned to more than one region: {len(conflicts)}")
        for c, cnt in sorted(conflicts.items())[:15]:
            L.append(f"  {c}: {cnt}")
    else:
        L.append("No country is assigned to more than one region.")
    L.append("")
    return "\n".join(L)
 
 
# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
 
def parse_args():
    p = argparse.ArgumentParser(description="Flag low-quality / suspicious nodes in a Nextstrain tree (no data is removed).")
    p.add_argument("--tree", required=True, help="Main Auspice tree JSON.")
    p.add_argument("--tip-freq", required=True, help="Tip-frequencies JSON.")
    p.add_argument("--output-dir", required=True, help="Folder for outputs (created if missing).")
    p.add_argument("--subtype", default="subtype", help="Label used in output file names (e.g. h1n1pdm).")
    p.add_argument("--qc-key", default="qc.overallStatus_ha", help="node_attrs key holding Nextclade QC status.")
    p.add_argument("--date-margin", type=float, default=0.25, help="Years of slack around the pivot range for tip dates.")
    p.add_argument("--max-date-ci-width", type=float, default=0.1, help="Tip date CI wider than this (years) is flagged imprecise.")
    p.add_argument("--parent-date-tol", type=float, default=0.01, help="Years a child may precede its parent before flagging.")
    p.add_argument("--long-branch-iqr", type=float, default=3.0, help="IQR multiplier for the long-branch Tukey fence.")
    p.add_argument("--freq-tol", type=float, default=1e-6, help="Tolerance when checking frequencies lie in [0, 1].")
    p.add_argument("--freq-sum-tol", type=float, default=0.01, help="Allowed deviation of per-pivot frequency totals from 1.")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print("Loading raw JSON files.")
    tree_json = load_json(args.tree)
    tip_json = load_json(args.tip_freq)
 
    print("Flattening tree.")
    records = flatten_tree(tree_json, args.qc_key)
    n_tips = sum(r["is_tip"] for r in records)
    print(f"           {len(records)} nodes found ({n_tips} tips, {len(records) - n_tips} internal).")
 
    print("Reading frequency file.")
    pivots, freqs = extract_frequencies(tip_json)
    print(f"           {len(pivots)} pivots, {len(freqs)} tip entries.")
 
    print("Variable type and QC checks")
    check_duplicate_names(records)
    check_qc_status(records)
    qc_dist = Counter(str(r["qc_status"]) for r in records if r["is_tip"])
    if args.qc_key not in set().union(*(r["_node_attrs"].keys() for r in records)):
        print(f"           WARNING: '{args.qc_key}' not found on any node; all tips will be flagged qc_missing.")
 
    print("Other attribute checks.")
    check_missing_and_placeholders(records)
    check_valid_numeric(records)
 
    print("Date and divergence checks.")
    check_dates(records, pivots, args.date_margin, args.max_date_ci_width, args.parent_date_tol)
    long_threshold = check_divergence(records, args.long_branch_iqr)
 
    print("Metadata checks.")
    check_ambiguous_mutations(records)
    conflicts = check_region_country_consistency(records)
    freq_findings = check_frequencies(records, pivots, freqs, args.freq_tol)
 
    print("Writing outputs.")
    finalize_flags(records)
    node_csv = out_dir / f"{args.subtype}_node_flags.csv"
    audit_csv = out_dir / f"{args.subtype}_attribute_audit.csv"
    report_txt = out_dir / f"{args.subtype}_flag_report.txt"
 
    write_node_table(records, node_csv)
    write_audit(attribute_audit(records), audit_csv)
    report = build_report(args.subtype, records, freq_findings, conflicts, long_threshold, args, qc_dist)
    report_txt.write_text(report + "\n")
 
    print()
    print(report)
    print()
    print(f"Wrote:\n  {node_csv}\n  {audit_csv}\n  {report_txt}")
 
 
if __name__ == "__main__":
    main()