"""Regression baselines on node features under the per-origin selection protocol.

Families (blocks): lbi, mom, nestar (H3N2, 2020+), mlr_mom (allele+mom), nodereg (node+mom+lbi),
full (node+mom+lbi+allele). Pooled (shared dense coefs, subtype-specific alleles) and per-subtype.
Usage: python3 run_regression.py
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import forecast_framework as ff
import selection as sel
from models.regression import Regression, available

TAU_PAIRS = [(0.25, 1.0), (0.5, 2.0), (0.25, 2.0)]
LAM_A = [0.01, 0.1, 1.0, 10.0]
LAM_F = [1e-3, 1e-2, 1e-1]
BASES = ["x", "pi"]
FAMILIES = {"lbi": ["lbi"], "mom": ["mom"], "nestar": ["nestar"], "mlr_mom": ["allele", "mom"],
            "nodereg": ["node", "mom", "lbi"], "full": ["node", "mom", "lbi", "allele"]}


def grid(blocks):
    g = []
    taus = TAU_PAIRS if ("mom" in blocks) else [(0.25, 1.0)]
    lam_as = LAM_A if "allele" in blocks else [None]
    for tau, la, lf, base in itertools.product(taus, lam_as, LAM_F, BASES):
        if base == "pi" and "mom" not in blocks and tau != TAU_PAIRS[0]: continue
        g.append(dict(blocks=blocks, tau=tau, lam_allele=la, lam_feat=lf, base=base))
    return g


def run(family, pool):
    blocks = FAMILIES[family]
    groups = [list(ff.SUBTYPES)] if pool == "pooled" else [[s] for s in ff.SUBTYPES]
    if family == "nestar": groups = [["h3n2"]]
    rows, chosen = [], []
    ok_fn = lambda cfg: (lambda k: available(cfg, k))
    for T in range(ff.FIRST_TEST_TARGET, ff.YEARS[-1] + 1):
        for subs in groups:
            test = [(s, T - 1) for s in subs if available({"blocks": blocks}, (s, T - 1))]
            if not test or not sel.val_origins(T, subs, ok_fn({"blocks": blocks})):
                continue
            scored = sel.score_configs(Regression, grid(blocks), T, subs, ok_fn)
            top = [c for _, c in scored[:sel.TOP]]
            fit_keys = sel.train_keys(subs, T - 1, ok_fn({"blocks": blocks}))
            preds = sel.ensemble_predict(Regression, top, fit_keys, test)
            for k in test:
                pr = ff.get_pair(*k)
                rows.append(dict(model=f"{family}_{pool}", pair=pr.name, subtype=k[0], t=k[1], emd=ff.emd(preds[k], pr.future.freqs, pr.D)))
            chosen.append(dict(T=T, subtypes=subs, val_emd=scored[0][0], top=[{kk: v for kk, v in c.items() if kk != "blocks"} for c in top]))
            print(f"  {family}_{pool} T={T} {'+'.join(subs)} val={scored[0][0]:.3f} test={[round(r['emd'], 3) for r in rows[-len(test):]]}", flush=True)
    return rows, chosen


if __name__ == "__main__":
    out = os.path.join(ff.ROOT, "results", "regression"); os.makedirs(out, exist_ok=True)
    fams = sys.argv[1:] or list(FAMILIES)
    for fam in fams:
        for pool in (["pooled"] if fam in ("lbi", "mom", "nestar") else ["pooled", "persub"]):
            rows, chosen = run(fam, pool)
            pd.DataFrame(rows).to_csv(os.path.join(out, f"{fam}_{pool}.csv"), index=False)
            json.dump(chosen, open(os.path.join(out, f"{fam}_{pool}_selection.json"), "w"), indent=1, default=str)
            print(f"{fam}_{pool}: mean test EMD {np.mean([r['emd'] for r in rows]):.3f} over {len(rows)} pairs", flush=True)
