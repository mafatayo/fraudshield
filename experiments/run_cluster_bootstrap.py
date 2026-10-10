"""
Cluster (block) bootstrap by destination account (Round 7, external examiner defect 3).
Refits XGBoost G4 and the random forest G4 on the temporal protocol (seed 42,
exactly as run_multiseed.py), then resamples DESTINATION ACCOUNTS (all test rows
with the same nameDest form one cluster) with replacement, 2,000 replicates.
Per replicate: PR-AUC, recall at the validation-F1 threshold, recall at zero
false positives (threshold recomputed inside the replicate), and the difference
in missed frauds (XGBoost minus random forest) at each model's validation-F1
threshold. Intervals are percentile 95 per cent intervals.
Output: results/cluster_bootstrap.json
"""
import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from r6_common import OUT, build_frame, feature_sets, temporal_masks

SEED = 42
NJ = 8
B = 2000
df, TYPE_COLS = build_frame()
SETS = feature_sets(TYPE_COLS)
X = df[SETS["G4_full_plus_velocity"]].to_numpy(np.float32)
y = df.isFraud.values.astype(np.int8)
tr, va, te = temporal_masks(df.step.values)
PAIR = len(sys.argv) > 1 and sys.argv[1] == "pair"
if PAIR:
    # In this PaySim file the fraud pairs are NOT linked by account identifiers
    # (no CASH_OUT nameOrig equals a TRANSFER nameDest), but a fraudulent TRANSFER and its
    # CASH_OUT share the same step and amount. Fraud rows are therefore clustered by
    # (step, amount); legitimate rows keep their destination account as cluster.
    dest = df.nameDest.values[te].astype(object)
    fi = np.flatnonzero(y[te] == 1)
    st = df.step.values[te][fi]; am = df.amount.values[te][fi]
    dest[fi] = ["F|%d|%r" % (a_, b_) for a_, b_ in zip(st, am)]
else:
    dest = df.nameDest.values[te]
yte = y[te]
cl, cid = np.unique(dest, return_inverse=True)
ncl = len(cl)


def val_f1_threshold(y_va, p_va):
    p, r, th = precision_recall_curve(y_va, p_va)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(th[np.nanargmax(f1s[:-1])])


spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))
clf = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
                        n_jobs=NJ, random_state=SEED, early_stopping_rounds=50)
clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
S = {"xgb": (clf.predict_proba(X[te])[:, 1].astype(np.float64),
             val_f1_threshold(y[va], clf.predict_proba(X[va])[:, 1]))}
print("xgb done", flush=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                            n_jobs=NJ, random_state=SEED)
rf.fit(X[tr], y[tr])
S["rf"] = (rf.predict_proba(X[te])[:, 1].astype(np.float64),
           val_f1_threshold(y[va], rf.predict_proba(X[va])[:, 1]))
print("rf done", flush=True)
del X

pos_mask = yte == 1
G = {}
for m, (s, thr) in S.items():
    u, inv = np.unique(s, return_inverse=True)
    inv = (len(u) - 1) - inv
    G[m] = (inv, len(u), s >= thr, s > s[~pos_mask].max())
    print(m, "point: PR-AUC", average_precision_score(yte, s), "missed@F1", int((pos_mask & (s < thr)).sum()),
          "missed@0FP", int((pos_mask & ~G[m][3]).sum()), flush=True)


def wap(pos_g, neg_g):
    ctp = np.cumsum(pos_g)
    cfp = np.cumsum(neg_g)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    rec = ctp / ctp[-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


per_cl = np.bincount(cid[pos_mask], minlength=ncl)
info = {"n_test_rows": int(len(yte)), "n_destination_clusters": int(ncl), "n_frauds": int(pos_mask.sum()),
        "n_clusters_containing_fraud": int((per_cl > 0).sum()),
        "frauds_per_fraud_cluster_mean": float(pos_mask.sum() / (per_cl > 0).sum()),
        "frauds_per_fraud_cluster_max": int(per_cl.max()),
        "clusters_with_two_or_more_frauds": int((per_cl >= 2).sum())}
print(info, flush=True)

rng = np.random.default_rng(20261007)
boot = {m: {"ap": [], "rec_f1": [], "rec_0fp": [], "miss_f1": []} for m in S}
for b in range(B):
    cnt = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)).astype(np.float64)
    w = cnt[cid]
    wp = w * pos_mask
    wn = w * ~pos_mask
    for m, (inv, ng, pred_f1, pred_0) in G.items():
        pg = np.bincount(inv, weights=wp, minlength=ng)
        ngr = np.bincount(inv, weights=wn, minlength=ng)
        tot = pg.sum()
        boot[m]["ap"].append(wap(pg, ngr))
        boot[m]["rec_f1"].append(float((wp * pred_f1).sum() / tot))
        first_neg = int(np.argmax(ngr > 0))  # groups are ordered from highest score
        boot[m]["rec_0fp"].append(float(pg[:first_neg].sum() / tot))
        boot[m]["miss_f1"].append(float((wp * ~pred_f1).sum()))
    if b % 200 == 0:
        print("rep", b, flush=True)


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


tf = json.load(open(os.path.join(OUT, "threshold_free.json")))["temporal"]["models"]["xgb"]
out = {"replicates": B, "seed": 20261007, "model_seed": SEED, "cluster_key": ("fraud rows by (step, amount) pair, legitimate rows by nameDest" if PAIR else "nameDest"), "cluster_info": info, "xgb": {}, "rf": {}}
for m in S:
    out[m] = {"pr_auc_point": float(average_precision_score(yte, S[m][0])),
              "pr_auc_ci95": ci(boot[m]["ap"]), "recall_val_f1_ci95": ci(boot[m]["rec_f1"]),
              "recall_zero_fp_ci95": ci(boot[m]["rec_0fp"]),
              "recall_val_f1_point": float((pos_mask & G[m][2]).sum() / pos_mask.sum()),
              "recall_zero_fp_point": float((pos_mask & G[m][3]).sum() / pos_mask.sum())}
x = out["xgb"]
x["poisson_row_bootstrap_ci95"] = {"pr_auc": tf["pr_auc_ci95"], "recall_val_f1": tf["recall_at_val_f1_threshold_ci95"],
                                   "recall_zero_fp": tf["recall_at_zero_fp_ci95"]}


def wd(c):
    return c[1] - c[0]


x["width_ratio_cluster_over_poisson"] = {
    "pr_auc": wd(x["pr_auc_ci95"]) / wd(tf["pr_auc_ci95"]),
    "recall_val_f1": wd(x["recall_val_f1_ci95"]) / wd(tf["recall_at_val_f1_threshold_ci95"]),
    "recall_zero_fp": wd(x["recall_zero_fp_ci95"]) / wd(tf["recall_at_zero_fp_ci95"])}
d = np.array(boot["xgb"]["miss_f1"]) - np.array(boot["rf"]["miss_f1"])
mc = json.load(open(os.path.join(OUT, "mcnemar.json")))["temporal"]["val_f1_thresholds"]
out["missed_frauds_difference_xgb_minus_rf_at_val_f1"] = {
    "point": int((pos_mask & ~G["xgb"][2]).sum()) - int((pos_mask & ~G["rf"][2]).sum()),
    "mean": float(d.mean()), "ci95": ci(d), "share_positive": float((d > 0).mean()),
    "share_zero": float((d == 0).mean()), "share_negative": float((d < 0).mean()),
    "existing_exact_mcnemar": {"p": mc["p_exact_two_sided"], "rf_only": mc["rf_only_catches"],
                               "xgb_only": mc["xgb_only_catches"]}}
json.dump(out, open(os.path.join(OUT, ("cluster_bootstrap_pairlinked.json" if PAIR else "cluster_bootstrap.json")), "w"), indent=1)
print("DONE", json.dumps(out, indent=1), flush=True)
