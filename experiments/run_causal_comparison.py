"""
Non-causal comparator for RQ2 (Chapter One): does the causal (prior-only)
construction of the velocity family cost anything relative to a naive,
leakage-permitting construction of the same features computed from each
account's FULL transaction history, including transactions after the one
being scored?

Builds a second, non-causal version of the three velocity features most
exposed to this distinction (destPriorTxnCount, destPriorAmtMean, and
amountOverDestMean, which depends on destPriorAmtMean -- swapped for a
full-history count and mean per destination account) and compares G4
(causal, full feature set) against G4_leaky (identical except for those
three features) under the same temporal protocol, same seed, same
hyperparameters, same column order.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

RNG = 42
DATA = r"C:\Users\VICTUS2025\.cache\kagglehub\datasets\ealaxi\paysim1\versions\2\PS_20174392719_1491204439457_log.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

df = pd.read_csv(DATA)
df = df.sort_values("step", kind="stable").reset_index(drop=True)

df["hour"] = df.step % 24
df["day"] = df.step // 24
df["errBalOrig"] = df.newbalanceOrig + df.amount - df.oldbalanceOrg
df["errBalDest"] = df.oldbalanceDest + df.amount - df.newbalanceDest
df["origZeroedOut"] = ((df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)).astype(np.int8)
df["destIsMerchant"] = df.nameDest.str.startswith("M").astype(np.int8)
df["amountToBalanceRatio"] = df.amount / (df.oldbalanceOrg + 1.0)

g_orig = df.groupby("nameOrig", sort=False)
df["origPriorTxnCount"] = g_orig.cumcount()
df["origTimeSinceLast"] = (df.step - g_orig["step"].shift(1)).fillna(-1)
g_dest = df.groupby("nameDest", sort=False)
df["destPriorTxnCount"] = g_dest.cumcount()
df["destTimeSinceLast"] = (df.step - g_dest["step"].shift(1)).fillna(-1)
prior_amt_sum = g_dest["amount"].cumsum() - df.amount
df["destPriorAmtMean"] = (prior_amt_sum / df.destPriorTxnCount.replace(0, np.nan)).fillna(0.0)
df["amountOverDestMean"] = df.amount / (df.destPriorAmtMean + 1.0)

# --- non-causal ("leaky") counterparts: full-history stats per destination
# account, computed using EVERY transaction that account is ever involved
# in, including ones after the row being scored ---
dest_full_count = g_dest["amount"].transform("count") - 1  # exclude self
dest_full_mean = (g_dest["amount"].transform("sum") - df.amount) / dest_full_count.replace(0, np.nan)
df["destPriorTxnCount_leaky"] = dest_full_count.fillna(0)
df["destPriorAmtMean_leaky"] = dest_full_mean.fillna(0.0)
df["amountOverDestMean_leaky"] = df.amount / (df.destPriorAmtMean_leaky + 1.0)

type_d = pd.get_dummies(df.type, prefix="type", dtype=np.int8)
df = pd.concat([df, type_d], axis=1)
TYPE_COLS = list(type_d.columns)

BASE = ["amount", "hour", "day", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
        "newbalanceDest", "errBalOrig", "errBalDest", "origZeroedOut", "destIsMerchant",
        "amountToBalanceRatio", "origPriorTxnCount", "origTimeSinceLast"]

FEATS_CAUSAL = BASE + ["destPriorTxnCount", "destTimeSinceLast", "destPriorAmtMean",
                        "amountOverDestMean"] + TYPE_COLS
FEATS_LEAKY = BASE + ["destPriorTxnCount_leaky", "destTimeSinceLast", "destPriorAmtMean_leaky",
                       "amountOverDestMean_leaky"] + TYPE_COLS

y = df.isFraud.values.astype(np.int8)
step = df.step.values
q60, q70 = np.quantile(step, [0.60, 0.70])
tr = step <= q60
va = (step > q60) & (step <= q70)
te = step > q70
spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))


def run(name, feats):
    X = df[feats].astype(np.float32)
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=50, n_jobs=-1, random_state=RNG)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    prob_va = clf.predict_proba(X[va])[:, 1]
    prob_te = clf.predict_proba(X[te])[:, 1]
    p, r, th = precision_recall_curve(y[va], prob_va)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    thr = float(th[np.nanargmax(f1s[:-1])])
    pred = (prob_te >= thr).astype(int)
    out = {
        "n_features": len(feats),
        "pr_auc": float(average_precision_score(y[te], prob_te)),
        "roc_auc": float(roc_auc_score(y[te], prob_te)),
        "precision": float(precision_score(y[te], pred)),
        "recall": float(recall_score(y[te], pred)),
        "f1": float(f1_score(y[te], pred)),
        "train_s": time.time() - t0,
        "best_iteration": int(clf.best_iteration),
    }
    print(name, out, flush=True)
    return out


results = {
    "G4_causal": run("G4_causal", FEATS_CAUSAL),
    "G4_noncausal_leaky": run("G4_noncausal_leaky", FEATS_LEAKY),
}
with open(os.path.join(OUT, "causal_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)
print("DONE", flush=True)
