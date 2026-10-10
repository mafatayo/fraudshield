"""
Rolling-origin evaluation (Round 7, response to external examiner defect 2).
Five origins: train end at the 50, 55, 60, 65, 70 per cent quantiles of step;
validation = next 10 per cent of the step range; test = following 20 per cent
(for the 70 per cent origin this is q80 to q100). XGBoost (G4, early stopping on
the validation window, parameters of run_multiseed.py) and the 100-tree depth-12
random forest on G4. One seed (42) per origin.
For the 60 per cent origin a second, open-ended test window (step > q70) is also
scored with the same fitted models; it is the existing temporal protocol and is
used as the reproduction check against results/multiseed.json.
Output: results/rolling_origin.json (written after every origin).
"""
import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from r6_common import OUT, build_frame, feature_sets

SEED = 42
NJ = 8
RESULT_PATH = os.path.join(OUT, "rolling_origin.json")
df, TYPE_COLS = build_frame()
SETS = feature_sets(TYPE_COLS)
FS = "G4_full_plus_velocity"
X = df[SETS[FS]].to_numpy(np.float32)
y = df.isFraud.values.astype(np.int8)
step = df.step.values


def val_f1_threshold(y_va, p_va):
    p, r, th = precision_recall_curve(y_va, p_va)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(th[np.nanargmax(f1s[:-1])])


def rab(neg_sorted, pos, k):
    thr = neg_sorted[k] if k < len(neg_sorted) else -np.inf
    return float((pos > thr).mean())


def evaluate(y_te, p_te, thr):
    pos = p_te[y_te == 1]
    neg = p_te[y_te == 0]
    ns = np.sort(neg)[::-1]
    n_neg = len(neg)
    n_pos = len(pos)
    pred = p_te >= thr
    tp = int((pred & (y_te == 1)).sum())
    fp = int((pred & (y_te == 0)).sum())
    return {
        "n_test": int(len(y_te)), "n_frauds": int(n_pos),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "thr_val_f1": thr, "missed_at_val_f1": n_pos - tp, "false_alarms_at_val_f1": fp,
        "recall_at_val_f1": tp / n_pos,
        "missed_at_zero_fp": int((pos <= ns[0]).sum()),
        "recall_at_zero_fp": float((pos > ns[0]).mean()),
        "recall_at_fpr_1e-5": rab(ns, pos, int(1e-5 * n_neg)),
        "recall_at_fpr_1e-4": rab(ns, pos, int(1e-4 * n_neg)),
        "recall_at_fpr_1e-3": rab(ns, pos, int(1e-3 * n_neg)),
    }


RES = {"config": {"seed": SEED, "feature_set": FS, "origins_pct": [50, 55, 60, 65, 70],
                  "windows": "train step<=q(p); val q(p)..q(p+10); test q(p+10)..q(p+30); quantiles of step",
                  "xgb": "n_estimators=600 max_depth=8 lr=0.1 subsample=0.8 colsample_bytree=0.8 min_child_weight=5 "
                         "scale_pos_weight=train ratio early_stopping=50 on validation aucpr",
                  "rf": "100 trees depth 12 class_weight=balanced_subsample"}, "origins": {}}
for pct in (50, 55, 60, 65, 70):
    t0 = time.time()
    qa, qb, qc = np.quantile(step, [pct / 100, (pct + 10) / 100, (pct + 30) / 100])
    tr = step <= qa
    va = (step > qa) & (step <= qb)
    te = (step > qb) & (step <= qc)
    node = {"step_cuts": [float(qa), float(qb), float(qc)],
            "frauds": {"train": int(y[tr].sum()), "validation": int(y[va].sum()), "test": int(y[te].sum())},
            "rows": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())}, "models": {}}
    wins = {"test": te}
    if pct == 60:
        wins["test_open_ended_reproduction"] = step > qb
        node["frauds"]["test_open_ended"] = int(y[step > qb].sum())
    ytr, yva = y[tr], y[va]
    spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
                            n_jobs=NJ, random_state=SEED, early_stopping_rounds=50)
    clf.fit(X[tr], ytr, eval_set=[(X[va], yva)], verbose=False)
    thr = val_f1_threshold(yva, clf.predict_proba(X[va])[:, 1])
    node["models"]["xgb"] = {"trees_used": int(clf.best_iteration) + 1}
    for wn, m in wins.items():
        node["models"]["xgb"][wn] = evaluate(y[m], clf.predict_proba(X[m])[:, 1], thr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                                n_jobs=NJ, random_state=SEED)
    rf.fit(X[tr], ytr)
    thr_r = val_f1_threshold(yva, rf.predict_proba(X[va])[:, 1])
    node["models"]["rf"] = {"trees_used": 100}
    for wn, m in wins.items():
        node["models"]["rf"][wn] = evaluate(y[m], rf.predict_proba(X[m])[:, 1], thr_r)
    node["fit_s"] = time.time() - t0
    RES["origins"][str(pct)] = node
    json.dump(RES, open(RESULT_PATH, "w"), indent=1)
    x = node["models"]["xgb"]["test"]
    r = node["models"]["rf"]["test"]
    print(f"[origin {pct}] frauds {node['frauds']} XGB PR={x['pr_auc']:.5f} miss@F1={x['missed_at_val_f1']} "
          f"miss@0FP={x['missed_at_zero_fp']} | RF PR={r['pr_auc']:.5f} miss@F1={r['missed_at_val_f1']} "
          f"miss@0FP={r['missed_at_zero_fp']} ({node['fit_s']:.0f}s)", flush=True)

keys = ["pr_auc", "roc_auc", "recall_at_zero_fp", "recall_at_fpr_1e-5", "recall_at_fpr_1e-4", "recall_at_fpr_1e-3",
        "missed_at_val_f1", "missed_at_zero_fp", "false_alarms_at_val_f1"]
RES["summary"] = {}
for mdl in ("xgb", "rf"):
    RES["summary"][mdl] = {}
    for k in keys:
        v = np.array([RES["origins"][str(p)]["models"][mdl]["test"][k] for p in (50, 55, 60, 65, 70)], float)
        RES["summary"][mdl][k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
json.dump(RES, open(RESULT_PATH, "w"), indent=1)
print("DONE", flush=True)
