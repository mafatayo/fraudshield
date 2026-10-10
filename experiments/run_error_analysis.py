"""
Error analysis of the temporal-protocol test window (seed 42, saved scores).
Output: results/error_analysis.json

Describes the frauds that the headline models miss at the validation-F1 threshold
(XGBoost 22, random forest 38), the overlap between the two sets, the two frauds that
both models miss even at the best single threshold, and how missed frauds differ from
caught frauds on type, amount, drain pattern and timing.
"""
import json
import os

import numpy as np

from r6_common import OUT, build_frame, temporal_masks

df, _ = build_frame()
M = json.load(open(os.path.join(OUT, "metrics.json")))
z = np.load(os.path.join(OUT, "curves_temporal.npz"))
tr, va, te = temporal_masks(df.step.values)
d = df[te].reset_index(drop=True)
y = z["y"].astype(int)
assert (d.isFraud.values == y).all(), "row order of curves_temporal.npz does not match the test window"
res = {"test_rows": int(len(d)), "frauds": int(y.sum())}


def describe(mask, label):
    s = d[mask]
    return {
        "n": int(len(s)),
        "TRANSFER": int((s.type == "TRANSFER").sum()), "CASH_OUT": int((s.type == "CASH_OUT").sum()),
        "median_amount": float(s.amount.median()) if len(s) else None,
        "min_amount": float(s.amount.min()) if len(s) else None,
        "max_amount": float(s.amount.max()) if len(s) else None,
        "share_amount_equals_oldbalanceOrg_pct": float(100 * (s.amount == s.oldbalanceOrg).mean()) if len(s) else None,
        "share_errBalOrig_zero_pct": float(100 * (s.errBalOrig.abs() < 1).mean()) if len(s) else None,
        "share_oldbalanceDest_zero_pct": float(100 * (s.oldbalanceDest == 0).mean()) if len(s) else None,
        "share_dest_has_prior_txn_pct": float(100 * (s.destPriorTxnCount > 0).mean()) if len(s) else None,
        "median_step": float(s.step.median()) if len(s) else None,
    }


miss = {}
for name, key in (("xgb", "xgb_prob"), ("rf", "rf_prob")):
    p = z[key].astype(np.float64)
    thr = M["temporal_split"][name]["threshold"]
    m = (y == 1) & (p < thr)
    miss[name] = m
    res[f"{name}_missed_at_val_f1"] = describe(m, name)
    neg_max = p[y == 0].max()
    m0 = (y == 1) & (p <= neg_max)
    miss[name + "_zero_fp"] = m0
    res[f"{name}_missed_at_zero_fp"] = describe(m0, name)
res["caught_by_both_at_val_f1"] = describe((y == 1) & ~miss["xgb"] & ~miss["rf"], "caught")
res["overlap"] = {
    "xgb_missed": int(miss["xgb"].sum()), "rf_missed": int(miss["rf"].sum()),
    "missed_by_both": int((miss["xgb"] & miss["rf"]).sum()),
    "missed_by_xgb_only": int((miss["xgb"] & ~miss["rf"]).sum()),
    "missed_by_rf_only": int((~miss["xgb"] & miss["rf"]).sum()),
}
both0 = miss["xgb_zero_fp"] & miss["rf_zero_fp"]
res["frauds_missed_by_both_even_at_zero_fp"] = [
    {"step": int(r.step), "type": r.type, "amount": float(r.amount), "oldbalanceOrg": float(r.oldbalanceOrg),
     "newbalanceOrig": float(r.newbalanceOrig), "oldbalanceDest": float(r.oldbalanceDest),
     "newbalanceDest": float(r.newbalanceDest)} for r in d[both0].itertuples()]
# frauds that are not full drains (the pattern the simple rules rely on)
nd = (y == 1) & ~(d.amount == d.oldbalanceOrg).values
res["frauds_not_full_drain"] = {"n": int(nd.sum()), "missed_by_xgb": int((nd & miss["xgb"]).sum()),
                                "missed_by_rf": int((nd & miss["rf"]).sum())}
json.dump(res, open(os.path.join(OUT, "error_analysis.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
