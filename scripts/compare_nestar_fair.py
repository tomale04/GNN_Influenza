"""Fair ne_star comparison on the 4 H3N2 test pairs it covers (t = 2022..2025 -> targets 2023..2026)."""
import os, glob
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAIRS = [f"h3n2 {t}->{t+1}" for t in range(2022, 2026)]

def load(paths, suffix=""):
    df = pd.concat([pd.read_csv(p) for p in paths])[["model", "pair", "emd"]]
    df["model"] = df["model"] + suffix
    return df[df.pair.isin(PAIRS)]

A = pd.concat([load([os.path.join(R, "linear_fitness.csv")]), load(glob.glob(os.path.join(R, "regression", "*.csv"))),
               load(glob.glob(os.path.join(R, "gnn_v2", "*.csv")))])
B = load(glob.glob(os.path.join(R, "regression_nestar_fair", "*.csv")), suffix=" [h3n2,t>=2020]")
B["model"] = B["model"].str.replace("_persub", "").str.replace("_pooled", "")
base = A[A.model.isin(["naive", "oracle_bound"])]
rng = np.random.default_rng(0); idx = rng.integers(0, 4, (10000, 4))

def table(df, title):
    W = df.pivot_table(index="pair", columns="model", values="emd").loc[PAIRS]
    ref_ne = [c for c in W.columns if c.startswith("nestar")][0]
    rows = []
    for m in W.columns:
        r = dict(model=m, mean=W[m].mean())
        for ref, lab in [("naive", "vs naive"), (ref_ne, "vs ne_star")]:
            d = (W[m] - W[ref]).values; b = d[idx].mean(1)
            r[lab] = f"{d.mean():+.2f} [{np.percentile(b, 2.5):+.2f},{np.percentile(b, 97.5):+.2f}] wins {int((d < 0).sum())}/4"
        rows.append(r)
    S = pd.DataFrame(rows).sort_values("mean")
    print(f"\n=== {title} ===\n" + S.round(3).to_string(index=False))
    print("\nper pair:\n" + W[S.model].T.round(2).to_string())
    return S

pd.set_option("display.width", 250)
SA = table(A, "(A) same 4 test pairs, models as originally trained")
SB = table(pd.concat([B, base]), "(B) same 4 test pairs AND same information (H3N2 only, training pairs t>=2020)")
pd.concat([SA.assign(view="A"), SB.assign(view="B")]).to_csv(os.path.join(R, "nestar_fair_comparison.csv"), index=False)
