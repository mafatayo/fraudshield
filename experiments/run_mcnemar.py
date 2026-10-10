"""
Exact McNemar test for the RF-versus-XGBoost comparison, computed from the
saved test-set probabilities (results/curves_*.npz) and the validation-F1
thresholds stored in results/metrics.json. Output: results/mcnemar.json

It reports the test at three operating points, because the result depends on
where each model's threshold falls:
  1. each model at its own validation-F1 threshold (the thesis protocol);
  2. each model at its own zero-false-positive ("oracle") threshold, the
     lowest cut-off at which no legitimate test transaction is flagged;
  3. both models at the same fixed threshold of 0.5.
The McNemar test conditions on discordant pairs among the frauds only
(transactions flagged by one model but not the other), exactly as in the
thesis; legitimate transactions contribute no discordance at points 1 and 2
because both models raise no false alarms there.
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
metrics = json.load(open(os.path.join(OUT, "metrics.json")))


def discordance(pred_a, pred_b, y):
    pos = y == 1
    a_only = int((pred_a & ~pred_b & pos).sum())
    b_only = int((~pred_a & pred_b & pos).sum())
    neg_a_only = int((pred_a & ~pred_b & ~pos).sum())
    neg_b_only = int((~pred_a & pred_b & ~pos).sum())
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"rf_only_catches": a_only, "xgb_only_catches": b_only, "p_exact_two_sided": p,
            "rf_only_false_alarms": neg_a_only, "xgb_only_false_alarms": neg_b_only}


out = {}
for tag, key in [("random", "random_split"), ("temporal", "temporal_split")]:
    z = np.load(os.path.join(OUT, f"curves_{tag}.npz"))
    y = z["y"]; rf = z["rf_prob"].astype(np.float64); xg = z["xgb_prob"].astype(np.float64)
    thr_rf = metrics[key]["rf"]["threshold"]; thr_xg = metrics[key]["xgb"]["threshold"]
    o_rf = rf[y == 0].max(); o_xg = xg[y == 0].max()
    node = {"n_frauds": int(y.sum()), "n_legit": int((y == 0).sum())}
    # NOTE: probabilities were stored as float32; the thesis thresholds were
    # applied to float64 probabilities, so a handful of rows exactly at a
    # threshold could differ. The counts below reproduce Table 4.2.
    node["val_f1_thresholds"] = {"rf": thr_rf, "xgb": thr_xg,
                                 "fn_rf": int(((rf < thr_rf) & (y == 1)).sum()),
                                 "fn_xgb": int(((xg < thr_xg) & (y == 1)).sum()),
                                 **discordance(rf >= thr_rf, xg >= thr_xg, y)}
    node["zero_fp_oracle_thresholds"] = {
        "rf_max_legit_score": float(o_rf), "xgb_max_legit_score": float(o_xg),
        "fn_rf": int(((rf <= o_rf) & (y == 1)).sum()), "fn_xgb": int(((xg <= o_xg) & (y == 1)).sum()),
        **discordance(rf > o_rf, xg > o_xg, y)}
    node["fixed_threshold_0.5"] = {
        "fn_rf": int(((rf < 0.5) & (y == 1)).sum()), "fp_rf": int(((rf >= 0.5) & (y == 0)).sum()),
        "fn_xgb": int(((xg < 0.5) & (y == 1)).sum()), "fp_xgb": int(((xg >= 0.5) & (y == 0)).sum()),
        **discordance(rf >= 0.5, xg >= 0.5, y)}
    out[tag] = node

with open(os.path.join(OUT, "mcnemar.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
