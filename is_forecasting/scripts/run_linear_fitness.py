"""Rolling evaluation of the site-wise linear fitness model fit by CE or EMD loss.

Per fold and subtype: choose lam on the validation pair (model fit on train pairs), refit on
train+val with the chosen lam, evaluate EMD on the test pair. Usage: python3 run_linear_fitness.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import forecast_framework as ff
from models.linear_fitness import LinearFitness

LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LOSSES = ["ce", "emd"]

rows, chosen = [], []
for fold in ff.folds():
    T = fold["test_target"]
    for s in ff.SUBTYPES:
        tr = [ff.get_pair(*k) for k in fold["train"] if k[0] == s]
        va = [ff.get_pair(*k) for k in fold["val"] if k[0] == s]
        te = [k for k in fold["test"] if k[0] == s]
        for loss in LOSSES:
            val = {}
            for lam in LAMS:
                m = LinearFitness(loss=loss, lam=lam).fit(tr)
                val[lam] = np.mean([r["emd"] for r in ff.evaluate(m.predict, [(p.subtype, p.t) for p in va])])
            lam = min(val, key=val.get)
            m = LinearFitness(loss=loss, lam=lam).fit(tr + va)
            r = ff.evaluate(m.predict, te)[0]
            fitted = [(p.subtype, p.t) for p in tr + va]
            r["train_emd"] = np.mean([x["emd"] for x in ff.evaluate(m.predict, fitted)])
            r["train_emd_naive"] = np.mean([x["emd"] for x in ff.evaluate(ff.naive, fitted)])
            rows.append(dict(model=f"linear_{loss}", lam=lam, val_emd=val[lam], **r))
            chosen.append(dict(T=T, subtype=s, loss=loss, val_by_lam={str(k): round(v, 3) for k, v in val.items()}, lam=lam, n_coef=len(m.coefficients())))
            print(f"T={T} {s:8s} {loss:3s} lam={lam:g} val_emd={val[lam]:.3f} train_emd={r['train_emd']:.3f} (naive {r['train_emd_naive']:.3f}) test_emd={r['emd']:.3f}", flush=True)
        r = ff.evaluate(ff.naive, te)[0]; rows.append(dict(model="naive", **r))
        r = ff.evaluate(ff.oracle, te)[0]; rows.append(dict(model="oracle_bound", **r))

df = pd.DataFrame(rows)
out = os.path.join(ff.ROOT, "results"); os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "linear_fitness.csv"), index=False)
json.dump(chosen, open(os.path.join(out, "linear_fitness_lambda.json"), "w"), indent=1)

pd.set_option("display.width", 200)
print("\nTest EMD per pair (AAs):")
print(df.pivot_table(index="pair", columns="model", values="emd").round(3).to_string())
print("\nMean +/- sd over test folds:")
print(df.groupby(["subtype", "model"]).emd.agg(["mean", "std"]).round(3).to_string())
print("\nPooled over subtypes:")
print(df.groupby("model").emd.agg(["mean", "std", "count"]).round(3).to_string())
nv = df[df.model == "naive"].set_index("pair").emd
for mname in ["linear_ce", "linear_emd"]:
    d = df[df.model == mname].set_index("pair").emd - nv
    print(f"{mname}: beats naive in {(d < 0).sum()}/{len(d)} test pairs; mean diff {d.mean():+.3f} AA")

rng = np.random.default_rng(0)
for mname in ["linear_ce", "linear_emd"]:
    d = (df[df.model == mname].set_index("pair").emd - nv).values
    boots = rng.choice(d, (10000, len(d))).mean(1)
    print(f"{mname}: paired bootstrap mean diff vs naive {d.mean():+.3f} AA, 95% CI [{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}], P(diff>=0)={np.mean(boots>=0):.3f}")
