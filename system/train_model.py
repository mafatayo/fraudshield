"""Train the deployment model on the temporal training window and store model + threshold.

Same hyperparameters as experiments/run_experiments.py. The booster is truncated to the early
stopping iteration, and the threshold is the validation-F1 threshold, as in Section 3.4.4.
"""
import hashlib
import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT + "/system")
from r6_common import build_frame, temporal_masks  # noqa: E402
from fraudshield.config import FEATURES, MODEL_DIR  # noqa: E402

df, _ = build_frame()
X = df[FEATURES].to_numpy(np.float32)
y = df.isFraud.values.astype(np.int8)
tr, va, te = temporal_masks(df.step.values)
spw = float((y[tr] == 0).sum() / y[tr].sum())
clf = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
                        early_stopping_rounds=50, n_jobs=-1, random_state=42)
clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
best = int(clf.best_iteration)
booster = clf.get_booster()[: best + 1]
pv = booster.inplace_predict(X[va])
p, r, th = precision_recall_curve(y[va], pv)
f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
thr = float(th[np.nanargmax(f1[:-1])])
os.makedirs(MODEL_DIR, exist_ok=True)
booster.save_model(os.path.join(MODEL_DIR, "xgb_deploy.json"))
ver = hashlib.sha256(open(os.path.join(MODEL_DIR, "xgb_deploy.json"), "rb").read()).hexdigest()[:12]
json.dump({"features": FEATURES, "threshold": thr, "best_iteration": best, "rounds": best + 1,
           "version": ver, "trained_on": "temporal train window (step <= 60th percentile)"},
          open(os.path.join(MODEL_DIR, "meta.json"), "w"), indent=1)
pt = booster.inplace_predict(X[te])
print("rounds", best + 1, "threshold", thr, "version", ver)
print("test fraud", int(y[te].sum()), "flagged", int((pt >= thr).sum()), "caught", int(((pt >= thr) & (y[te] == 1)).sum()),
      "false alarms", int(((pt >= thr) & (y[te] == 0)).sum()))
