"""Cache the event stream (stable step order) and the offline scores of the deployed model."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT + "/system")
from r6_common import build_frame  # noqa: E402
from fraudshield.config import FEATURES, RUN_DIR  # noqa: E402
from fraudshield.scorer import Scorer  # noqa: E402

os.makedirs(RUN_DIR, exist_ok=True)
df, _ = build_frame()
sl = df[["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
         "oldbalanceDest", "newbalanceDest", "isFraud"]].copy()
sl.to_pickle(os.path.join(RUN_DIR, "stream.pkl"))
sc = Scorer()
X = df[FEATURES].to_numpy(np.float32)
s = sc.score_batch(X)
np.save(os.path.join(RUN_DIR, "offline_scores.npy"), s)
print(len(sl), "events;", int((s >= sc.threshold).sum()), "offline flags")
