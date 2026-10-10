"""
Round-6 multi-seed, threshold-free and pre-authorisation analysis.

For each seed this script refits, on the temporal protocol (and, for the
headline models, on the stratified random protocol):

  * the five nested ablation groups G1-G5 and the non-causal comparator
    (as in run_ablation.py / run_causal_comparison.py);
  * three PRE-AUTHORISATION feature sets that use only fields known before
    the transaction posts (no newbalanceOrig, newbalanceDest, errBalOrig,
    errBalDest, origZeroedOut);
  * G4 without the simulation-clock index `day`;
  * XGBoost with early stopping disabled (fixed 200 rounds), for G1, G4, G5
    and PRE_full, because the original runs stopped after one to eight
    trees on a saturated validation PR-AUC;
  * Random Forest (100 trees, depth 12) on G4 and PRE_full;
  * Logistic Regression and the headline models on the random protocol.

Per model it records threshold-free metrics (PR-AUC, ROC-AUC, recall at zero
false positives, recall at fixed false-positive-rate budgets) AND the
thresholded numbers used in the thesis (validation-F1 threshold), plus a
validation-midpoint threshold rule, so that the sensitivity of the missed-fraud
counts to the threshold rule is visible. Discordant-pair counts and exact
McNemar p-values are stored for the comparisons the thesis makes.

Seed 42 reproduces results/ablation.json and results/causal_comparison.json
for the shared configurations (check_seed42_reproduction() below).

Output: results/multiseed.json (written after every configuration, resumable).
Usage:  python run_multiseed.py [seed ...]      (default seeds 42..46)
"""
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from r6_common import OUT, build_frame, feature_sets, temporal_masks

RESULT_PATH = os.path.join(OUT, "multiseed.json")
SEEDS = [int(a) for a in sys.argv[1:]] or [42, 43, 44, 45, 46]

df, TYPE_COLS = build_frame()
SETS = feature_sets(TYPE_COLS)
y = df.isFraud.values.astype(np.int8)
step = df.step.values
tr_t, va_t, te_t = temporal_masks(step)
idx = np.arange(len(df))

if os.path.exists(RESULT_PATH):
    RES = json.load(open(RESULT_PATH))
else:
    RES = {"seeds": {}}


def save():
    with open(RESULT_PATH, "w") as f:
        json.dump(RES, f, indent=1)


def val_f1_threshold(y_va, p_va):
    p, r, th = precision_recall_curve(y_va, p_va)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(th[np.nanargmax(f1s[:-1])])


def val_midpoint_threshold(y_va, p_va):
    """Midpoint of the zero-error interval on validation when it exists,
    otherwise the validation-F1 threshold."""
    neg_max = p_va[y_va == 0].max()
    pos_min = p_va[y_va == 1].min()
    if pos_min > neg_max:
        return float((neg_max + pos_min) / 2), True
    return val_f1_threshold(y_va, p_va), False


def recall_at_fp_budget(neg_sorted_desc, pos, k):
    """Recall when at most k legitimate rows may be flagged (strict > rule)."""
    thr = neg_sorted_desc[k] if k < len(neg_sorted_desc) else -np.inf
    return float((pos > thr).mean())


def evaluate(y_te, p_te, y_va, p_va):
    pos = p_te[y_te == 1]
    neg = p_te[y_te == 0]
    neg_sorted = np.sort(neg)[::-1]
    thr = val_f1_threshold(y_va, p_va)
    thr_mid, mid_separable = val_midpoint_threshold(y_va, p_va)
    pred = p_te >= thr
    pred_mid = p_te >= thr_mid
    tp = int((pred & (y_te == 1)).sum()); fp = int((pred & (y_te == 0)).sum())
    tp_m = int((pred_mid & (y_te == 1)).sum()); fp_m = int((pred_mid & (y_te == 0)).sum())
    n_pos = int(len(pos)); n_neg = int(len(neg))
    out = {
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "thr_val_f1": thr, "tp": tp, "fp": fp, "fn": n_pos - tp,
        "recall": tp / n_pos, "precision": (tp / (tp + fp)) if tp + fp else 1.0,
        "thr_val_mid": thr_mid, "val_separable": bool(mid_separable),
        "tp_mid": tp_m, "fp_mid": fp_m, "fn_mid": n_pos - tp_m,
        "max_neg_score": float(neg_sorted[0]),
        "fn_at_zero_fp": int((pos <= neg_sorted[0]).sum()),
        "recall_at_zero_fp": float((pos > neg_sorted[0]).mean()),
        "recall_at_fpr_1e-5": recall_at_fp_budget(neg_sorted, pos, int(1e-5 * n_neg)),
        "recall_at_fpr_1e-4": recall_at_fp_budget(neg_sorted, pos, int(1e-4 * n_neg)),
        "recall_at_fpr_1e-3": recall_at_fp_budget(neg_sorted, pos, int(1e-3 * n_neg)),
        "n_unique_fraud_scores": int(len(np.unique(pos))),
        "fraud_score_median": float(np.median(pos)),
        "fraud_score_max": float(pos.max()),
    }
    return out, pred


def mcnemar_pair(pred_a, pred_b, y_te):
    pos = y_te == 1
    a_only = int((pred_a & ~pred_b & pos).sum())  # A catches, B misses
    b_only = int((~pred_a & pred_b & pos).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n > 0 else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_exact": float(p)}


def xgb_fit(Xtr, ytr, Xva, yva, seed, spw, fixed_rounds=None):
    kw = dict(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
              tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed)
    if fixed_rounds:
        kw.update(n_estimators=fixed_rounds)
        clf = xgb.XGBClassifier(**kw)
        clf.fit(Xtr, ytr, verbose=False)
        return clf, fixed_rounds
    clf = xgb.XGBClassifier(early_stopping_rounds=50, **kw)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf, int(clf.best_iteration) + 1


def log(msg):
    print(msg, flush=True)


def run_temporal_seed(seed):
    key = str(seed)
    node = RES["seeds"].setdefault(key, {})
    tmp = node.setdefault("temporal", {})
    preds = {}
    spw = float((y[tr_t] == 0).sum() / max(y[tr_t].sum(), 1))
    ytr, yva, yte = y[tr_t], y[va_t], y[te_t]

    plan = [
        ("xgb", "PRE_full", None), ("xgb", "PRE_noVel", None), ("xgb", "PRE_raw_velocity", None),
        ("xgb", "G4_full_plus_velocity", None), ("xgb", "G3_plus_consistency", None),
        ("xgb", "G1_transaction_only", None), ("xgb", "G2_plus_raw_balances", None),
        ("xgb", "G5_txn_plus_velocity_no_balance", None), ("xgb", "G4_noncausal_leaky", None),
        ("xgb", "G4_noDay", None),
        ("xgb", "G1_transaction_only", 200), ("xgb", "G4_full_plus_velocity", 200),
        ("xgb", "G5_txn_plus_velocity_no_balance", 200), ("xgb", "PRE_full", 200),
        ("rf", "G4_full_plus_velocity", None), ("rf", "PRE_full", None),
    ]
    for kind, fs, fixed in plan:
        name = f"{kind}:{fs}" + (f":fixed{fixed}" if fixed else "")
        need_pred = fs in ("PRE_full", "PRE_noVel", "G4_full_plus_velocity", "G3_plus_consistency",
                           "G4_noncausal_leaky") and not fixed
        if name in tmp and not need_pred:
            continue
        t0 = time.time()
        X = df[SETS[fs]].to_numpy(np.float32)
        Xtr, Xva, Xte = X[tr_t], X[va_t], X[te_t]
        if kind == "xgb":
            clf, ntrees = xgb_fit(Xtr, ytr, Xva, yva, seed, spw, fixed)
            if fixed:
                p_va = clf.predict_proba(Xva)[:, 1]; p_te = clf.predict_proba(Xte)[:, 1]
            else:
                p_va = clf.predict_proba(Xva)[:, 1]; p_te = clf.predict_proba(Xte)[:, 1]
            extra = {"trees_used": ntrees, "trees_stored": int(clf.get_booster().num_boosted_rounds())}
        else:
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=seed)
            rf.fit(Xtr, ytr)
            p_va = rf.predict_proba(Xva)[:, 1]; p_te = rf.predict_proba(Xte)[:, 1]
            extra = {"trees_used": 100}
        m, pred = evaluate(yte, p_te, yva, p_va)
        m.update(extra); m["n_features"] = len(SETS[fs]); m["fit_s"] = time.time() - t0
        preds[name] = pred
        if name not in tmp:
            tmp[name] = m
            save()
        log(f"[seed {seed} temporal] {name}: PR-AUC={m['pr_auc']:.5f} ROC={m['roc_auc']:.6f} "
            f"FN@val-F1={m['fn']} FP={m['fp']} FN@0FP={m['fn_at_zero_fp']} "
            f"trees={extra['trees_used']} ({time.time()-t0:.0f}s)")
        del X, Xtr, Xva, Xte
    # discordant-pair comparisons at each model's own validation-F1 threshold
    pairs = {
        "G3_vs_G4": ("xgb:G3_plus_consistency", "xgb:G4_full_plus_velocity"),
        "G4_vs_leaky": ("xgb:G4_full_plus_velocity", "xgb:G4_noncausal_leaky"),
        "PREfull_vs_G4": ("xgb:PRE_full", "xgb:G4_full_plus_velocity"),
        "PREfull_vs_PREnoVel": ("xgb:PRE_full", "xgb:PRE_noVel"),
        "RFG4_vs_XGBG4": ("rf:G4_full_plus_velocity", "xgb:G4_full_plus_velocity"),
        "RFPRE_vs_XGBPRE": ("rf:PRE_full", "xgb:PRE_full"),
    }
    node["temporal_pairs"] = node.get("temporal_pairs", {})
    for pname, (a, b) in pairs.items():
        if a in preds and b in preds:
            node["temporal_pairs"][pname] = mcnemar_pair(preds[a], preds[b], yte)
    save()


def run_random_seed(seed):
    key = str(seed)
    node = RES["seeds"].setdefault(key, {})
    rnd = node.setdefault("random", {})
    tr, tmpi = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmpi, test_size=2 / 3, stratify=y[tmpi], random_state=seed)
    ytr, yva, yte = y[tr], y[va], y[te]
    spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    for kind, fs in [("xgb", "G4_full_plus_velocity"), ("xgb", "PRE_full"), ("rf", "G4_full_plus_velocity"),
                     ("lr", "G4_full_plus_velocity")]:
        name = f"{kind}:{fs}"
        if name in rnd:
            continue
        t0 = time.time()
        X = df[SETS[fs]].to_numpy(np.float32)
        Xtr, Xva, Xte = X[tr], X[va], X[te]
        if kind == "xgb":
            clf, ntrees = xgb_fit(Xtr, ytr, Xva, yva, seed, spw)
            p_va = clf.predict_proba(Xva)[:, 1]; p_te = clf.predict_proba(Xte)[:, 1]
            extra = {"trees_used": ntrees}
        elif kind == "rf":
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=seed)
            rf.fit(Xtr, ytr)
            p_va = rf.predict_proba(Xva)[:, 1]; p_te = rf.predict_proba(Xte)[:, 1]
            extra = {"trees_used": 100}
        else:
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
            lr.fit(sc.transform(Xtr), ytr)
            p_va = lr.predict_proba(sc.transform(Xva))[:, 1]; p_te = lr.predict_proba(sc.transform(Xte))[:, 1]
            extra = {}
        m, _ = evaluate(yte, p_te, yva, p_va)
        m.update(extra); m["fit_s"] = time.time() - t0
        rnd[name] = m
        save()
        log(f"[seed {seed} random] {name}: PR-AUC={m['pr_auc']:.5f} FN@val-F1={m['fn']} FP={m['fp']} "
            f"FN@0FP={m['fn_at_zero_fp']} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    RES["config"] = {"seeds": SEEDS, "xgb": "n_estimators=600 max_depth=8 lr=0.1 subsample=0.8 "
                     "colsample_bytree=0.8 min_child_weight=5 scale_pos_weight=train ratio early_stopping=50",
                     "rf": "100 trees depth 12 class_weight=balanced_subsample"}
    for s in SEEDS:
        run_temporal_seed(s)
        run_random_seed(s)
    save()
    log("DONE")
