"""
Descriptive statistics quoted in Chapters 1 to 4 that earlier drafts stated
without a script behind them. Output: results/consistency_stats.json

  1. errBalOrig / errBalDest shares among TRANSFER and CASH_OUT events, by
     class, under three equality definitions (exact zero, |x| < 0.01,
     |x| < 1.00).
  2. The all-legitimate classifier's accuracy.
  3. Fraud prevalence by 100-step block (class-prior drift) and by phase.
  4. How often origin accounts repeat (what the origin-side velocity
     features can see) and destination-side history by class.
  5. Hand-written rule baselines on the full data and on the temporal test
     window, including the post-posting rule that adds errBalOrig == 0, and a
     pre-authorisation drain rule that uses only the pre-transaction balance.
  6. isFlaggedFraud facts (how often it fires relative to large transfers).
"""
import json
import os

import numpy as np
import pandas as pd

from r6_common import DATA, OUT, build_frame, temporal_masks

df, _ = build_frame()
n = len(df)
y = df.isFraud.values
res = {"rows": int(n), "frauds": int(y.sum())}

res["fraud_rate"] = float(y.mean())
res["all_legitimate_accuracy"] = float(1 - y.mean())

# 1. residual shares ----------------------------------------------------------
tc = df[df.type.isin(["TRANSFER", "CASH_OUT"])]
fr, lg = tc[tc.isFraud == 1], tc[tc.isFraud == 0]
res["tc_counts"] = {"fraud": int(len(fr)), "legit": int(len(lg))}
res["residual_shares"] = {}
for label, tol in [("exact", 0.0), ("tol_0.01", 0.01), ("tol_1.00", 1.0)]:
    def zero(s):
        return (s.abs() <= tol) if tol == 0.0 else (s.abs() < tol)
    res["residual_shares"][label] = {
        "errBalOrig_zero_fraud_pct": float(100 * zero(fr.errBalOrig).mean()),
        "errBalOrig_nonzero_legit_pct": float(100 * (~zero(lg.errBalOrig)).mean()),
        "errBalDest_nonzero_fraud_pct": float(100 * (~zero(fr.errBalDest)).mean()),
        "errBalDest_nonzero_legit_pct": float(100 * (~zero(lg.errBalDest)).mean()),
    }

# 3. prevalence by step block --------------------------------------------------
blocks = []
for lo in range(1, 744, 100):
    hi = min(lo + 99, 743)
    m = (df.step >= lo) & (df.step <= hi)
    steps = hi - lo + 1
    blocks.append({
        "steps": f"{lo}-{hi}", "rows": int(m.sum()), "frauds": int(y[m].sum()),
        "fraud_prevalence_pct": float(100 * y[m].mean()),
        "frauds_per_step": float(y[m].sum() / steps),
        "legit_per_step": float(((y[m] == 0).sum()) / steps),
    })
res["prevalence_by_100_step_block"] = blocks
res["prevalence_ratio_last_to_first_block"] = blocks[-1]["fraud_prevalence_pct"] / blocks[0]["fraud_prevalence_pct"]
tr, va, te = temporal_masks(df.step.values)
res["prevalence_pct"] = {"temporal_train": float(100 * y[tr].mean()), "temporal_val": float(100 * y[va].mean()),
                         "temporal_test": float(100 * y[te].mean()), "overall": float(100 * y.mean())}
res["horizon"] = {"steps": 743, "days": 743 / 24}

# 4. repeat accounts ------------------------------------------------------------
res["origin_accounts"] = {
    "unique": int(df.nameOrig.nunique()),
    "rows_with_prior_origin_txn_pct": float(100 * (df.origPriorTxnCount > 0).mean()),
    "fraud_rows_with_prior_origin_txn_pct": float(100 * (df.origPriorTxnCount[y == 1] > 0).mean()),
}
res["dest_history_pct_with_prior_txn"] = {
    "all_rows": float(100 * (df.destPriorTxnCount > 0).mean()),
    "fraud": float(100 * (df.destPriorTxnCount[y == 1] > 0).mean()),
    "legit_transfer_cashout": float(100 * (lg.destPriorTxnCount > 0).mean()),
}

# 5. rule baselines ---------------------------------------------------------------
tc_mask = df.type.isin(["TRANSFER", "CASH_OUT"])
rules = {
    "R0_isFlaggedFraud": df.isFlaggedFraud.values == 1,
    "R1_drain_newbalance_zero_amount_ge_99pct": (tc_mask & (df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)
                                                 & (df.amount >= df.oldbalanceOrg * 0.99)).values,
    "R2_post_posting_drain_and_errBalOrig_zero": (tc_mask & (df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)
                                                  & (df.errBalOrig == 0)).values,
    "R3_pre_auth_drain_amount_ge_oldbalance": (tc_mask & (df.oldbalanceOrg > 0)
                                               & (df.amount >= df.oldbalanceOrg)).values,
    "R4_pre_auth_drain_and_first_time_destination": (tc_mask & (df.oldbalanceOrg > 0)
                                                     & (df.amount >= df.oldbalanceOrg)
                                                     & (df.destPriorTxnCount == 0)).values,
}
res["rules"] = {}
for name, flag in rules.items():
    entry = {}
    for scope, mask in [("full_data", np.ones(n, bool)), ("temporal_test", te)]:
        f, yy = flag[mask], y[mask]
        tp = int((f & (yy == 1)).sum()); fp = int((f & (yy == 0)).sum())
        entry[scope] = {"flagged": int(f.sum()), "tp": tp, "fp": fp, "fn": int(yy.sum() - tp),
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(int(yy.sum()), 1)}
    res["rules"][name] = entry
res["share_of_fraud_with_amount_equal_oldbalance_pct"] = float(
    100 * ((df.amount == df.oldbalanceOrg) & (y == 1)).sum() / y.sum())

# 6. isFlaggedFraud -------------------------------------------------------------------
fl = df[df.isFlaggedFraud == 1]
res["isFlaggedFraud"] = {
    "n_flagged": int(len(fl)), "types": fl.type.value_counts().to_dict(),
    "min_amount": float(fl.amount.min()),
    "transfers_over_200000": int(((df.type == "TRANSFER") & (df.amount > 200000)).sum()),
}

with open(os.path.join(OUT, "consistency_stats.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
