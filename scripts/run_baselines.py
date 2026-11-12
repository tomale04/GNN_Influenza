"""Report naive / uniform / oracle-bound forecasts for every season pair and per rolling test fold."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import forecast_framework as ff

# sanity checks of the objective machinery
for s in ff.SUBTYPES:
    pr = ff.get_pair(s, 2022)
    assert abs(ff.emd(pr.target, pr.future.freqs, pr.D) - ff.emd_lower_bound(pr.D, pr.future.freqs)) < 1e-6
    assert abs(ff.emd(pr.future.freqs, pr.future.freqs, ff.hamming(pr.future.X, pr.future.X))) < 1e-9

all_pairs = [(s, t) for s in ff.SUBTYPES for t in ff.YEARS[:-1]]
rows = []
for name, fn in [("naive", ff.naive), ("uniform", ff.uniform), ("oracle_bound", ff.oracle)]:
    for r in ff.evaluate(fn, all_pairs):
        rows.append(dict(model=name, **r))
df = pd.DataFrame(rows)
df["split"] = np.where(df.t + 1 >= ff.FIRST_TEST_TARGET, "test", "train/val only")

wide = df.pivot_table(index=["subtype", "t", "split", "n_current", "n_future"], columns="model", values="emd").reset_index()
wide["pair"] = wide.subtype + " " + wide.t.astype(str) + "->" + (wide.t + 1).astype(str)
wide["naive_ce"] = df[df.model == "naive"].set_index(["subtype", "t"]).loc[list(zip(wide.subtype, wide.t)), "ce"].values
wide = wide[["pair", "split", "n_current", "n_future", "naive", "uniform", "oracle_bound", "naive_ce"]]
pd.set_option("display.width", 200)
print("EMD to the future (amino acids), per season pair:\n")
print(wide.round(3).to_string(index=False))

test = df[df.split == "test"]
summ = test.groupby(["model", "subtype"]).emd.agg(["mean", "std", "count"]).round(3)
print("\nRolling test folds (targets %d-%d), mean +/- sd EMD:\n" % (ff.FIRST_TEST_TARGET, ff.YEARS[-1]))
print(summ.to_string())

out = os.path.join(ff.ROOT, "results"); os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "baselines.csv"), index=False)
print("\nfolds:")
for f in ff.folds():
    print(f"  test target {f['test_target']}: train t in {sorted({t for _, t in f['train']})}, val t={f['val'][0][1]}, test t={f['test'][0][1]}")
