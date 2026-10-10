"""
Additional baselines requested by the Round-6 panel. Output: results/baselines_extra.json

  * LR_log1p: logistic regression on the 23 features after a signed log1p
    transform of the heavy-tailed monetary columns (amount, balances,
    residuals), standardised, balanced class weights. Tests whether the weak
    linear result is partly a feature-scaling artefact.
  * DT_depth3: a single decision tree of depth 3 (class_weight balanced).
  * DT_depth6: a single decision tree of depth 6.
  * single_feature_errBalOrig: rank by |errBalOrig| inverted (score = -|errBalOrig|),
    threshold chosen on validation F1, to show what one engineered column buys.
Both temporal and stratified-random protocols, seed 42, thresholds chosen on
validation F1 as in run_experiments.py. Threshold-free zero-FP recall is also
reported.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from r6_common import OUT, RNG, build_frame, feature_sets, temporal_masks

df, TYPE_COLS = build_frame()
FS = feature_sets(TYPE_COLS)["G4_full_plus_velocity"]
X = df[FS].to_numpy(np.float64)
y = df.isFraud.values.astype(np.int8)
step = df.step.values
MONEY = [i for i, c in enumerate(FS) if c in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                                              "newbalanceDest", "errBalOrig", "errBalDest", "destPriorAmtMean")]
Xlog = X.copy()
Xlog[:, MONEY] = np.sign(Xlog[:, MONEY]) * np.log1p(np.abs(Xlog[:, MONEY]))
i_err = FS.index("errBalOrig")


def evaluate(yte, pte, yva, pva):
    p, r, th = precision_recall_curve(yva, pva)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    thr = th[np.nanargmax(f1[:-1])]
    pred = pte >= thr
    tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum())
    pos = pte[yte == 1]; neg_max = pte[yte == 0].max()
    return {"pr_auc": float(average_precision_score(yte, pte)), "roc_auc": float(roc_auc_score(yte, pte)),
            "tp": tp, "fp": fp, "fn": int(yte.sum() - tp), "precision": tp / max(tp + fp, 1),
            "recall": tp / int(yte.sum()), "fn_at_zero_fp": int((pos <= neg_max).sum())}


idx = np.arange(len(df))
tr_t, va_t, te_t = temporal_masks(step)
trr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=RNG)
var, ter = train_test_split(tmp, test_size=2 / 3, stratify=y[tmp], random_state=RNG)
protocols = {"temporal": (np.flatnonzero(tr_t), np.flatnonzero(va_t), np.flatnonzero(te_t)),
             "random": (trr, var, ter)}
out = {}
for tag, (tr, va, te) in protocols.items():
    res = {}
    sc = StandardScaler().fit(Xlog[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(Xlog[tr]), y[tr])
    res["LR_log1p"] = evaluate(y[te], lr.predict_proba(sc.transform(Xlog[te]))[:, 1],
                               y[va], lr.predict_proba(sc.transform(Xlog[va]))[:, 1])
    for d in (3, 6):
        dt = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RNG)
        dt.fit(X[tr], y[tr])
        res[f"DT_depth{d}"] = evaluate(y[te], dt.predict_proba(X[te])[:, 1], y[va], dt.predict_proba(X[va])[:, 1])
    s = -np.abs(X[:, i_err])
    res["single_feature_errBalOrig"] = evaluate(y[te], s[te], y[va], s[va])
    out[tag] = res
    print(tag, json.dumps(res, indent=1), flush=True)
json.dump(out, open(os.path.join(OUT, "baselines_extra.json"), "w"), indent=1)
