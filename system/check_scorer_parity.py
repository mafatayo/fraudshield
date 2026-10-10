"""Compare the in-process tree evaluator with XGBoost on all 6,362,620 events (offline features).
Output: results/system_scorer_parity.json"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT + "/system")
from r6_common import build_frame  # noqa: E402
from fraudshield.config import FEATURES  # noqa: E402
from fraudshield.scorer import Scorer  # noqa: E402

df, _ = build_frame()
X = df[FEATURES].to_numpy(np.float32)
sc = Scorer()
ref = sc.score_batch(X)
n = len(X)
# evaluate the loop scorer on every row would take minutes; do all rows in a vectorised copy of the same loop
mine = np.empty(n, np.float32)
t0 = time.time()
m = np.full(n, sc.base_margin, dtype=np.float32)
for feat, thr, yes, no, val in sc.trees:
    node = np.zeros(n, dtype=np.int64)
    feat_a, thr_a, yes_a, no_a, val_a = (np.array(a) for a in (feat, thr, yes, no, val))
    thr_a = thr_a.astype(np.float32); val_a = val_a.astype(np.float32)
    while True:
        f = feat_a[node]
        leaf = f < 0
        if leaf.all():
            break
        xv = X[np.arange(n), np.where(leaf, 0, f)]
        nxt = np.where(xv < thr_a[node], yes_a[node], no_a[node])
        node = np.where(leaf, node, nxt)
    m = (m + val_a[node]).astype(np.float32)
mine = 1.0 / (1.0 + np.exp(-m))
d = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
# and the scalar loop, on a sample, against the same reference
idx = np.random.default_rng(0).choice(n, 200000, replace=False)
sd = np.array([abs(sc.score(X[i]) - float(ref[i])) for i in idx])
flags_ref = ref >= sc.threshold
out = {"rows": n, "vectorised_max_abs_diff": float(d.max()), "vectorised_rows_not_bit_equal": int((mine != ref).sum()),
       "flag_disagreements": int(((mine >= sc.threshold) != flags_ref).sum()),
       "scalar_sample_rows": len(idx), "scalar_sample_max_abs_diff": float(sd.max()),
       "scalar_sample_rows_over_1e-6": int((sd > 1e-6).sum())}
json.dump(out, open(os.path.join(ROOT, "results", "system_scorer_parity.json"), "w"), indent=1)
print(out)
