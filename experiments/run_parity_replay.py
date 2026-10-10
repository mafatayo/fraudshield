"""
Train/serve parity and causality tests for the six stateful features.
Output: results/parity_replay.json

Test 1 (replay parity). Stream all 6,362,620 events, in the stable step order
used for training, through the incremental dictionary-based updater
(incremental_features.py), reading state before each event and updating it
after, and compare each feature with the offline pandas column.

Test 2 (micro-batch skew). Repeat the replay with micro-batch semantics, in
which every event of one simulation step reads the state as it stood at the
start of that step (events in the same step cannot see each other), and count
the rows whose features then differ from the offline values. This is the
deviation that vectorised micro-batch processing (Section 3.2.2) would
introduce for accounts that appear twice in one step.

Test 3 (prefix invariance). Recompute the offline features on data truncated
at step t and check that the values for every row with step <= t equal the
values computed on the full data. Equality for all t is the operational
definition of "uses no information from the future" for these columns.

Test 4 (leaky comparator). The same prefix test applied to the deliberately
non-causal destination features of run_causal_comparison.py must FAIL,
which shows that Test 3 can detect a look-ahead construction.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from incremental_features import IncrementalState
from r6_common import OUT, build_frame

FEATS = ["origPriorTxnCount", "origTimeSinceLast", "destPriorTxnCount",
         "destTimeSinceLast", "destPriorAmtMean", "amountOverDestMean"]

df, _ = build_frame()
n = len(df)
step = df.step.values
amount = df.amount.values
orig = pd.factorize(df.nameOrig)[0]
dest = pd.factorize(df.nameDest)[0]
off = {f: df[f].to_numpy(np.float64) for f in FEATS}
res = {"rows": int(n)}

# ---- Test 1: sequential replay
t0 = time.time()
st = IncrementalState()
inc = np.empty((n, 6))
for i in range(n):
    s, a, o, d = step[i], amount[i], orig[i], dest[i]
    inc[i] = st.features(s, a, o, d)
    st.update(s, a, o, d)
res["replay_seconds"] = time.time() - t0
t1 = {}
for j, f in enumerate(FEATS):
    diff = np.abs(inc[:, j] - off[f])
    exact = int((diff == 0).sum())
    tol = int((diff <= 1e-6 * np.maximum(1.0, np.abs(off[f]))).sum())
    t1[f] = {"rows_exactly_equal": exact, "rows_equal_within_1e-6_relative": tol,
             "max_abs_diff": float(diff.max())}
res["test1_replay_parity"] = t1
res["test1_all_rows_equal_within_tolerance"] = bool(all(v["rows_equal_within_1e-6_relative"] == n for v in t1.values()))

# ---- Test 2: micro-batch (per-step) semantics
t0 = time.time()
mb = np.empty((n, 6))
bounds = np.flatnonzero(np.diff(step)) + 1
starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [n]])
st = IncrementalState()
for a_, b_ in zip(starts, ends):
    for i in range(a_, b_):
        mb[i] = st.features(step[i], amount[i], orig[i], dest[i])  # state frozen at step start
    for i in range(a_, b_):
        st.update(step[i], amount[i], orig[i], dest[i])
res["microbatch_seconds"] = time.time() - t0
differs = np.zeros(n, bool)
per_feat = {}
for j, f in enumerate(FEATS):
    d = np.abs(mb[:, j] - off[f]) > 1e-6 * np.maximum(1.0, np.abs(off[f]))
    per_feat[f] = int(d.sum()); differs |= d
y = df.isFraud.values
te_mask = step > np.quantile(step, 0.70)
res["test2_microbatch_skew"] = {
    "rows_with_any_feature_different": int(differs.sum()),
    "share_of_rows_pct": float(100 * differs.mean()),
    "fraud_rows_different": int((differs & (y == 1)).sum()),
    "share_of_fraud_pct": float(100 * (differs & (y == 1)).sum() / y.sum()),
    "share_of_temporal_test_rows_pct": float(100 * differs[te_mask].mean()),
    "per_feature_rows_different": per_feat,
}

# ---- Test 3 and 4: prefix invariance, using the pandas definitions
def offline_cols(sub):
    g_o = sub.groupby("nameOrig", sort=False); g_d = sub.groupby("nameDest", sort=False)
    out = pd.DataFrame(index=sub.index)
    out["origPriorTxnCount"] = g_o.cumcount()
    out["origTimeSinceLast"] = (sub.step - g_o["step"].shift(1)).fillna(-1)
    out["destPriorTxnCount"] = g_d.cumcount()
    out["destTimeSinceLast"] = (sub.step - g_d["step"].shift(1)).fillna(-1)
    ps = g_d["amount"].cumsum() - sub.amount
    out["destPriorAmtMean"] = (ps / out.destPriorTxnCount.replace(0, np.nan)).fillna(0.0)
    out["amountOverDestMean"] = sub.amount / (out.destPriorAmtMean + 1.0)
    return out

def leaky_cols(sub):
    g_d = sub.groupby("nameDest", sort=False)
    c = g_d["amount"].transform("count") - 1
    m = (g_d["amount"].transform("sum") - sub.amount) / c.replace(0, np.nan)
    return pd.DataFrame({"destPriorTxnCount_leaky": c.fillna(0), "destPriorAmtMean_leaky": m.fillna(0.0)},
                        index=sub.index)

base = df[["step", "amount", "nameOrig", "nameDest"]]
t3 = []
for t in (100, 300, 500, 700):
    keep = base.step <= t
    sub = base[keep]
    got = offline_cols(sub)
    ok = all(np.allclose(got[f].to_numpy(), df.loc[sub.index, f].to_numpy(), rtol=1e-9, atol=0) for f in FEATS)
    gl = leaky_cols(sub)
    full_leaky = df.loc[sub.index, ["destPriorTxnCount_leaky", "destPriorAmtMean_leaky"]]
    changed = int((gl.destPriorTxnCount_leaky.to_numpy() != full_leaky.destPriorTxnCount_leaky.to_numpy()).sum())
    t3.append({"truncate_at_step": t, "rows": int(keep.sum()),
               "causal_features_unchanged": bool(ok),
               "leaky_destPriorTxnCount_rows_changed_by_truncation": changed})
    print(t3[-1], flush=True)
res["test3_test4_prefix_invariance"] = t3

with open(os.path.join(OUT, "parity_replay.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
