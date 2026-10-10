"""
Threshold-free comparison and bootstrap confidence intervals for the saved
seed-42 test-set scores (results/curves_random.npz, curves_temporal.npz).
Output: results/threshold_free.json

Contents per protocol and model (LR, RF, XGBoost):
  * PR-AUC (average precision) and ROC-AUC with 95% Poisson-bootstrap CIs
    (2,000 replicates; each test row receives an independent Poisson(1)
    weight, tie-handling by grouping identical scores);
  * recall at zero false positives and at false-positive budgets;
  * a threshold sweep (TP/FP/FN at 0.3, 0.5, 0.7, 0.9 and the validation-F1
    threshold);
  * paired bootstrap difference RF minus XGBoost in PR-AUC and in
    recall at the validation-F1 thresholds and at zero false positives;
  * the width of the XGBoost "plateau": how many frauds are missed as the
    threshold moves between 0.60 and 0.85 (temporal protocol).
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
metrics = json.load(open(os.path.join(OUT, "metrics.json")))
rng = np.random.default_rng(20260101)
B = 2000


def groups(scores):
    """Return (group id per row ordered from highest score), n groups."""
    u, inv = np.unique(scores, return_inverse=True)
    inv = (len(u) - 1) - inv          # group 0 = highest score
    return inv, len(u)


def ap_and_auc_from_group_counts(pos_g, neg_g):
    """AP (step-wise, as sklearn) and ROC-AUC from per-group weighted counts."""
    ctp = np.cumsum(pos_g); cfp = np.cumsum(neg_g)
    P = ctp[-1]; N = cfp[-1]
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / P
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap = float(np.sum((rec - rec_prev) * prec))
    tpr = np.concatenate([[0.0], ctp / P]); fpr = np.concatenate([[0.0], cfp / N])
    auc = float(np.trapezoid(tpr, fpr))
    return ap, auc


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def point_stats(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": int(y.sum() - tp)}


out = {"bootstrap_replicates": B}
for tag, key in [("random", "random_split"), ("temporal", "temporal_split")]:
    z = np.load(os.path.join(OUT, f"curves_{tag}.npz"))
    y = z["y"].astype(np.int8)
    node = {"n_frauds": int(y.sum()), "n_legit": int((y == 0).sum()), "models": {}}
    model_scores = {"lr": z["lr_prob"].astype(np.float64), "rf": z["rf_prob"].astype(np.float64),
                    "xgb": z["xgb_prob"].astype(np.float64)}
    G = {m: groups(s) for m, s in model_scores.items()}
    weights = [rng.poisson(1.0, size=len(y)).astype(np.float32) for _ in range(0)]  # placeholder
    boot = {m: {"ap": [], "auc": [], "rec_f1": [], "rec_0fp": []} for m in model_scores}
    thr_f1 = {m: metrics[key][m]["threshold"] for m in model_scores}
    zero_fp = {m: model_scores[m][y == 0].max() for m in model_scores}
    for b in range(B):
        w = rng.poisson(1.0, size=len(y)).astype(np.float64)
        wp = w * (y == 1); wn = w * (y == 0)
        for m, s in model_scores.items():
            inv, ng = G[m]
            pos_g = np.bincount(inv, weights=wp, minlength=ng)
            neg_g = np.bincount(inv, weights=wn, minlength=ng)
            ap, auc = ap_and_auc_from_group_counts(pos_g, neg_g)
            boot[m]["ap"].append(ap); boot[m]["auc"].append(auc)
            boot[m]["rec_f1"].append(float((wp * (s >= thr_f1[m])).sum() / wp.sum()))
            boot[m]["rec_0fp"].append(float((wp * (s > zero_fp[m])).sum() / wp.sum()))
    for m, s in model_scores.items():
        pos = s[y == 1]; neg = np.sort(s[y == 0])[::-1]
        n_neg = len(neg)
        def rec_budget(k):
            return float((pos > neg[k]).mean())
        node["models"][m] = {
            "pr_auc": metrics[key][m]["pr_auc"], "pr_auc_ci95": ci(boot[m]["ap"]),
            "roc_auc": metrics[key][m]["roc_auc"], "roc_auc_ci95": ci(boot[m]["auc"]),
            "val_f1_threshold": point_stats(y, s, thr_f1[m]),
            "recall_at_val_f1_threshold_ci95": ci(boot[m]["rec_f1"]),
            "zero_fp_threshold_max_legit_score": float(zero_fp[m]),
            "fn_at_zero_fp": int((pos <= zero_fp[m]).sum()),
            "recall_at_zero_fp": float((pos > zero_fp[m]).mean()),
            "recall_at_zero_fp_ci95": ci(boot[m]["rec_0fp"]),
            "recall_at_fpr_1e-5": rec_budget(int(1e-5 * n_neg)),
            "recall_at_fpr_1e-4": rec_budget(int(1e-4 * n_neg)),
            "recall_at_fpr_1e-3": rec_budget(int(1e-3 * n_neg)),
            "threshold_sweep": {str(t): point_stats(y, s, t) for t in (0.3, 0.5, 0.7, 0.9)},
            "fraud_score_median": float(np.median(pos)), "fraud_score_max": float(pos.max()),
            "n_unique_fraud_scores": int(len(np.unique(pos))),
        }
    # paired differences RF - XGB
    d_ap = np.array(boot["rf"]["ap"]) - np.array(boot["xgb"]["ap"])
    d_auc = np.array(boot["rf"]["auc"]) - np.array(boot["xgb"]["auc"])
    d_rf1 = np.array(boot["rf"]["rec_f1"]) - np.array(boot["xgb"]["rec_f1"])
    d_r0 = np.array(boot["rf"]["rec_0fp"]) - np.array(boot["xgb"]["rec_0fp"])
    node["paired_difference_rf_minus_xgb"] = {
        "pr_auc": {"mean": float(d_ap.mean()), "ci95": ci(d_ap)},
        "roc_auc": {"mean": float(d_auc.mean()), "ci95": ci(d_auc)},
        "recall_at_val_f1_thresholds": {"mean": float(d_rf1.mean()), "ci95": ci(d_rf1)},
        "recall_at_zero_fp": {"mean": float(d_r0.mean()), "ci95": ci(d_r0)},
    }
    # XGBoost plateau
    sx = model_scores["xgb"]; px = sx[y == 1]
    node["xgb_threshold_plateau"] = [
        {"threshold": float(t), "fn": int((px < t).sum()), "fp": int(((sx >= t) & (y == 0)).sum())}
        for t in np.arange(0.60, 0.8501, 0.01)]
    sr = model_scores["rf"]; pr_ = sr[y == 1]
    node["rf_threshold_grid"] = [
        {"threshold": float(t), "fn": int((pr_ < t).sum()), "fp": int(((sr >= t) & (y == 0)).sum())}
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88, 0.9)]
    out[tag] = node
    print(tag, "done", flush=True)

with open(os.path.join(OUT, "threshold_free.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
