"""Compare all models on the rolling test pairs (EMD, AAs) with paired bootstrap vs naive and vs linear_emd."""
import os, glob
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
lin = pd.read_csv(os.path.join(R, "linear_fitness.csv"))[["model", "pair", "subtype", "emd"]]
import sys
GNN_DIR = sys.argv[1] if len(sys.argv) > 1 else "gnn_v2"
gnn = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, GNN_DIR, "*.csv")))])[["model", "pair", "subtype", "emd"]]
df = pd.concat([lin, gnn])
W = df.pivot_table(index="pair", columns="model", values="emd")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(W), (10000, len(W)))


def boot(d):
    b = d[idx].mean(1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), (b >= 0).mean()


rows = []
for m in W.columns:
    r = dict(model=m, pooled=W[m].mean(), h3n2=W.loc[W.index.str.startswith("h3n2"), m].mean(),
             h1n1pdm=W.loc[W.index.str.startswith("h1n1pdm"), m].mean(), wins_vs_naive=int((W[m] < W["naive"]).sum()))
    for ref in ("naive", "linear_emd"):
        mu, lo, hi, p = boot((W[m] - W[ref]).values)
        r[f"d_{ref}"] = f"{mu:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}"
    rows.append(r)
S = pd.DataFrame(rows).sort_values("pooled")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(S.round(3).to_string(index=False))
print("\nPer pair:\n", W.round(2).to_string())
S.to_csv(os.path.join(R, f"model_comparison_{GNN_DIR}.csv"), index=False)
