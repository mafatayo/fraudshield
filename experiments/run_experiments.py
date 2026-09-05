"""
FraudShield journal-paper experiments on the PaySim dataset.

Trains rule-based / LR / RF / XGBoost fraud detectors under two evaluation
protocols (stratified random split and strict temporal split), with causal
behavioral/velocity features that mirror what the streaming layer (Kafka +
Spark Streaming + Redis) would maintain online. Saves all metrics to JSON
and publication figures to ../figures.
"""
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = 42
DATA = r"C:\Users\VICTUS2025\.cache\kagglehub\datasets\ealaxi\paysim1\versions\2\PS_20174392719_1491204439457_log.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
})

results = {"env": {
    "python": platform.python_version(),
    "cpu": platform.processor(),
    "machine": platform.machine(),
    "xgboost": xgb.__version__,
}}

# ---------------------------------------------------------------- load
t0 = time.time()
df = pd.read_csv(DATA)
print(f"loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)

results["dataset"] = {
    "rows": int(len(df)),
    "frauds": int(df.isFraud.sum()),
    "fraud_rate": float(df.isFraud.mean()),
    "steps": [int(df.step.min()), int(df.step.max())],
    "type_counts": df.type.value_counts().to_dict(),
    "fraud_by_type": df.groupby("type").isFraud.sum().to_dict(),
}

# Rule-based baseline = the simulator's built-in business rule (isFlaggedFraud)
y_all = df.isFraud.values
flag = df.isFlaggedFraud.values
tn, fp, fn, tp = confusion_matrix(y_all, flag).ravel()
results["rule_baseline_full"] = {
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "precision": float(tp / max(tp + fp, 1)),
    "recall": float(tp / max(tp + fn, 1)),
}
print("rule baseline:", results["rule_baseline_full"], flush=True)

# ---------------------------------------------------------------- features
t0 = time.time()
# stable causal ordering: by step, then original arrival order within step
df = df.sort_values("step", kind="stable").reset_index(drop=True)

df["hour"] = df.step % 24
df["day"] = df.step // 24

# balance-consistency features (streaming-computable from the event itself)
df["errBalOrig"] = df.newbalanceOrig + df.amount - df.oldbalanceOrg
df["errBalDest"] = df.oldbalanceDest + df.amount - df.newbalanceDest
df["origZeroedOut"] = ((df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)).astype(np.int8)
df["destIsMerchant"] = df.nameDest.str.startswith("M").astype(np.int8)
df["amountToBalanceRatio"] = df.amount / (df.oldbalanceOrg + 1.0)

# causal per-account state features (what Redis holds online): built strictly
# from each account's PRIOR transactions
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
FEATURES = (["amount", "hour", "day", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "errBalOrig", "errBalDest",
             "origZeroedOut", "destIsMerchant", "amountToBalanceRatio",
             "origPriorTxnCount", "origTimeSinceLast", "destPriorTxnCount",
             "destTimeSinceLast", "destPriorAmtMean", "amountOverDestMean"]
            + list(type_d.columns))
X_full = pd.concat([df[FEATURES[:18]], type_d], axis=1).astype(np.float32)
y_full = df.isFraud.values.astype(np.int8)
step = df.step.values
print(f"features built in {time.time()-t0:.1f}s -> {X_full.shape}", flush=True)
results["features"] = FEATURES


def eval_model(name, y_true, prob, y_val, prob_val):
    """Tune threshold on validation (max F1), report test metrics."""
    p, r, th = precision_recall_curve(y_val, prob_val)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    best = np.nanargmax(f1s[:-1])
    thr = float(th[best])
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    m = {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "threshold": thr,
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    print(f"  {name}: PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}", flush=True)
    return m


def run_protocol(tag, tr, va, te):
    print(f"\n=== protocol: {tag} ===", flush=True)
    Xtr, ytr = X_full.iloc[tr], y_full[tr]
    Xva, yva = X_full.iloc[va], y_full[va]
    Xte, yte = X_full.iloc[te], y_full[te]
    out = {"sizes": {"train": len(tr), "val": len(va), "test": len(te)},
           "fraud_counts": {"train": int(ytr.sum()), "val": int(yva.sum()),
                            "test": int(yte.sum())}}
    spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    out["scale_pos_weight"] = spw

    # Logistic Regression
    t0 = time.time()
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(sc.transform(Xtr), ytr)
    out["lr_train_s"] = time.time() - t0
    out["lr"] = eval_model("LR", yte, lr.predict_proba(sc.transform(Xte))[:, 1],
                           yva, lr.predict_proba(sc.transform(Xva))[:, 1])

    # Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=RNG)
    rf.fit(Xtr, ytr)
    out["rf_train_s"] = time.time() - t0
    out["rf"] = eval_model("RF", yte, rf.predict_proba(Xte)[:, 1],
                           yva, rf.predict_proba(Xva)[:, 1])

    # XGBoost
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=50, n_jobs=-1, random_state=RNG)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    out["xgb_train_s"] = time.time() - t0
    out["xgb_best_iteration"] = int(clf.best_iteration)
    prob_te = clf.predict_proba(Xte)[:, 1]
    prob_va = clf.predict_proba(Xva)[:, 1]
    out["xgb"] = eval_model("XGB", yte, prob_te, yva, prob_va)

    # curves for figures
    p, r, _ = precision_recall_curve(yte, prob_te)
    fpr, tpr, _ = roc_curve(yte, prob_te)
    np.savez(os.path.join(OUT, f"curves_{tag}.npz"),
             pr_p=p, pr_r=r, fpr=fpr, tpr=tpr,
             lr_prob=lr.predict_proba(sc.transform(Xte))[:, 1].astype(np.float32),
             rf_prob=rf.predict_proba(Xte)[:, 1].astype(np.float32),
             xgb_prob=prob_te.astype(np.float32), y=yte)
    return out, clf


# ------------------------------------------------- protocol A: random split
idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y_full, random_state=RNG)
va, te = train_test_split(tmp, test_size=2 / 3, stratify=y_full[tmp], random_state=RNG)
results["random_split"], xgb_rand = run_protocol("random", tr, va, te)

# ------------------------------------------------ protocol B: temporal split
cum = np.searchsorted(np.sort(step), step, side="right") / len(step)
q60, q70 = np.quantile(step, [0.60, 0.70])
tr = idx[step <= q60]
va = idx[(step > q60) & (step <= q70)]
te = idx[step > q70]
results["temporal_split_steps"] = {"train_max_step": float(q60), "val_max_step": float(q70)}
results["temporal_split"], xgb_temp = run_protocol("temporal", tr, va, te)

# ------------------------------------------------- feature importance (gain)
imp = xgb_rand.get_booster().get_score(importance_type="gain")
fmap = {f"f{i}": n for i, n in enumerate(X_full.columns)}
imp_named = {fmap.get(k, k): v for k, v in imp.items()}
results["xgb_gain_importance"] = dict(sorted(imp_named.items(), key=lambda kv: -kv[1]))

# ------------------------------------------------- latency microbenchmark
print("\nlatency benchmark...", flush=True)
booster = xgb_rand.get_booster()
sample = X_full.sample(20000, random_state=RNG).values.astype(np.float32)

lat_single = []
for i in range(5000):
    row = sample[i:i + 1]
    t0 = time.perf_counter()
    booster.inplace_predict(row)
    lat_single.append((time.perf_counter() - t0) * 1000)
lat_single = np.array(lat_single[500:])  # drop warmup

lat_batch = []
for i in range(200):
    chunk = sample[np.random.randint(0, 20000, 1000)]
    t0 = time.perf_counter()
    booster.inplace_predict(chunk)
    lat_batch.append((time.perf_counter() - t0) * 1000)
lat_batch = np.array(lat_batch[20:])

results["latency_ms"] = {
    "single_p50": float(np.percentile(lat_single, 50)),
    "single_p95": float(np.percentile(lat_single, 95)),
    "single_p99": float(np.percentile(lat_single, 99)),
    "single_mean": float(lat_single.mean()),
    "batch1000_p50": float(np.percentile(lat_batch, 50)),
    "batch1000_p99": float(np.percentile(lat_batch, 99)),
    "throughput_txn_per_s_batched": float(1000 / (np.percentile(lat_batch, 50) / 1000)),
}
np.savez(os.path.join(OUT, "latency.npz"), single=lat_single, batch=lat_batch)
print(results["latency_ms"], flush=True)

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
xgb_rand.get_booster().save_model(os.path.join(OUT, "xgb_random.json"))
xgb_temp.get_booster().save_model(os.path.join(OUT, "xgb_temporal.json"))
print("\nDONE — metrics.json written", flush=True)
