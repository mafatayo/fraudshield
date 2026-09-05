"""Feature-group ablation for FraudShield (XGBoost, temporal split)."""
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

type_d = pd.get_dummies(df.type, prefix="type", dtype=np.int8)
df = pd.concat([df, type_d], axis=1)

TXN = ["amount", "hour", "day"] + list(type_d.columns)
BAL = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CONS = ["errBalOrig", "errBalDest", "origZeroedOut", "destIsMerchant", "amountToBalanceRatio"]
VEL = ["origPriorTxnCount", "origTimeSinceLast", "destPriorTxnCount",
       "destTimeSinceLast", "destPriorAmtMean", "amountOverDestMean"]

GROUPS = {
    "G1_transaction_only": TXN,
    "G2_plus_raw_balances": TXN + BAL,
    "G3_plus_consistency": TXN + BAL + CONS,
    "G4_full_plus_velocity": TXN + BAL + CONS + VEL,
    "G5_txn_plus_velocity_no_balance": TXN + VEL,
}

y = df.isFraud.values.astype(np.int8)
step = df.step.values
q60, q70 = np.quantile(step, [0.60, 0.70])
tr = step <= q60
va = (step > q60) & (step <= q70)
te = step > q70
spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))

out = {}
for name, feats in GROUPS.items():
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
    out[name] = {
        "n_features": len(feats),
        "pr_auc": float(average_precision_score(y[te], prob_te)),
        "roc_auc": float(roc_auc_score(y[te], prob_te)),
        "precision": float(precision_score(y[te], pred)),
        "recall": float(recall_score(y[te], pred)),
        "f1": float(f1_score(y[te], pred)),
        "train_s": time.time() - t0,
        "best_iteration": int(clf.best_iteration),
    }
    print(name, out[name], flush=True)

with open(os.path.join(OUT, "ablation.json"), "w") as f:
    json.dump(out, f, indent=2)
print("DONE", flush=True)
