"""Per-origin selection protocol (mlr_mom_K.pdf Sec. 7.1), adapted to our rolling folds.

For test target T (test pair t = T-1 -> T):
  - validation origins: the K_VAL most recent t_v <= T-2 that have >= 1 earlier training pair;
    each scored by a fit on pairs with t + 1 <= t_v only;
  - each config scored by mean EMD over all validation pairs;
  - the forecasts of the TOP best configs, each refit on all pairs with t + 1 <= T - 1, are averaged.
"""
import numpy as np
import forecast_framework as ff

K_VAL, TOP = 3, 5


def train_keys(subtypes, t_max_target, ok=lambda k: True):
    return [(s, t) for s in subtypes for t in ff.YEARS[:-1] if t + 1 <= t_max_target and ok((s, t))]


def val_origins(T, subtypes, ok=lambda k: True):
    out = []
    for tv in range(T - 2, ff.YEARS[0] - 1, -1):
        vk = [(s, tv) for s in subtypes if ok((s, tv))]
        if vk and train_keys(subtypes, tv, ok):
            out.append((tv, vk))
        if len(out) == K_VAL: break
    return out


def score_configs(make_model, grid, T, subtypes, ok_fn=lambda cfg: (lambda k: True)):
    scores = []
    for cfg in grid:
        ok = ok_fn(cfg)
        emds = []
        for tv, vk in val_origins(T, subtypes, ok):
            m = make_model(cfg).fit(train_keys(subtypes, tv, ok))
            for k in vk:
                pr = ff.get_pair(*k)
                emds.append(ff.emd(m.predict(k), pr.future.freqs, pr.D))
        scores.append((float(np.mean(emds)) if emds else np.inf, cfg))
    return sorted(scores, key=lambda x: x[0])


def ensemble_predict(make_model, cfgs, fit_keys, pred_keys):
    """Average forecasts of cfgs, each fit on fit_keys; returns {key: probs}."""
    preds = {k: 0.0 for k in pred_keys}
    for cfg in cfgs:
        m = make_model(cfg).fit(fit_keys)
        for k in pred_keys: preds[k] = preds[k] + m.predict(k) / len(cfgs)
    return preds
