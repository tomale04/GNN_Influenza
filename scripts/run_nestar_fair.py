"""Fair comparison with the ne_star baseline (H3N2 only; ne_star exists in trees 2020+).

(B) refits every regression family under ne_star's constraints: H3N2 only, training/validation pairs t >= 2020,
same per-origin selection protocol, test targets 2023-2026. Writes results/regression_nestar_fair/*.csv.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import forecast_framework as ff
import selection as sel
import run_regression as rr
from models import regression as reg

rr.FAMILIES["nodereg_nestar"] = ["node", "mom", "lbi", "nestar"]
_orig = reg.available
reg.available = lambda cfg, key: key[0] == "h3n2" and key[1] >= 2020 and _orig(cfg, key)
rr.available = reg.available

out = os.path.join(ff.ROOT, "results", "regression_nestar_fair"); os.makedirs(out, exist_ok=True)
for fam in ["nestar", "lbi", "mom", "mlr_mom", "nodereg", "full", "nodereg_nestar"]:
    rr.FAMILIES[fam]
    orig_run_groups = None
    rows, chosen = rr.run(fam, "persub")  # persub -> groups [[h3n2],[h1n1pdm]]; h1n1pdm filtered out by available()
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{fam}.csv"), index=False)
    json.dump(chosen, open(os.path.join(out, f"{fam}_selection.json"), "w"), indent=1, default=str)
    print(f"{fam}: mean test EMD {np.mean([r['emd'] for r in rows]):.3f} over {len(rows)} pairs", flush=True)
