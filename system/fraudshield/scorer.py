"""Gradient-boosted tree scorer.

The deployed booster is a few shallow trees, so scoring is a handful of comparisons. The trees are
read straight from the XGBoost model file and evaluated by a small loop. That avoids both the roughly
1 ms of Python call overhead that Booster.inplace_predict adds to a one-row call and a 10 to 15 s
import of the xgboost package each time a worker restarts after a crash.
results/system_scorer_parity.json (system/check_scorer_parity.py) compares this evaluator with XGBoost
on all 6,362,620 events.
"""
import json
import math
import os

import numpy as np

from .config import MODEL_DIR


class Scorer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.meta = json.load(open(os.path.join(model_dir, "meta.json")))
        self.threshold = self.meta["threshold"]
        self.version = self.meta["version"]
        model = json.load(open(os.path.join(model_dir, "xgb_deploy.json")))["learner"]
        base = float(model["learner_model_param"]["base_score"].strip("[]"))
        self.base_margin = np.float32(math.log(base / (1.0 - base)))
        self.trees = []
        for t in model["gradient_booster"]["model"]["trees"]:
            left, right = t["left_children"], t["right_children"]
            feat = [-1 if left[i] == -1 else t["split_indices"][i] for i in range(len(left))]
            thr = [np.float32(v) for v in t["split_conditions"]]   # a leaf stores its value here
            self.trees.append((feat, thr, left, right, thr))

    def margin(self, x):
        m = self.base_margin
        for feat, thr, yes, no, val in self.trees:
            i = 0
            while feat[i] >= 0:
                i = yes[i] if x[feat[i]] < thr[i] else no[i]
            m = np.float32(m + val[i])
        return m

    def score(self, x):
        return float(1.0 / (1.0 + np.exp(-np.float32(self.margin(x)))))

    def score_batch(self, X):
        """XGBoost's own prediction, used only by the offline reference and the parity check."""
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(os.path.join(self.model_dir, "xgb_deploy.json"))
        return b.inplace_predict(X)
