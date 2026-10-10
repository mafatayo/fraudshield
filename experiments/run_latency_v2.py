"""
Round-6 latency and throughput characterisation. Output: results/latency_v2.json

Fixes the scope problems of the original benchmark (run_experiments.py), which
timed the random-split booster (56 stored trees, best iteration 5) with all
CPU threads on a pre-built feature matrix. This script reports, for explicit
models, thread counts and workloads:

  A. single-event latency (p50/p95/p99/max) for
       T8   temporal XGBoost, best-iteration trees only (8 trees)
       T58  temporal XGBoost, all 58 stored trees
       R56  random-split XGBoost, all 56 stored trees (the original benchmark)
       X600 an XGBoost model forced to 600 rounds (no early stopping), the
            configured ceiling, trained here on the temporal training window
       RF   Random Forest, 100 trees of depth 12 (sklearn predict_proba)
     each with the default thread pool and with a single thread, repeated five
     times with run-to-run spread;
  B. batch-of-1,000 latency and the derived batch rate, again repeated;
  C. SUSTAINED runs of 30 s each (single-event loop and batch loop), reporting
     the achieved rate over the whole interval, not a rate derived from one
     median;
  D. end-to-end in-process scoring on 20,000 temporal-test events, including
     dictionary-state feature construction (stand-in for the Redis reads and
     writes, with no network), timed per event, and a check that the scores
     equal those from the pre-built feature matrix.
"""
import json
import os
import platform
import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from incremental_features import IncrementalState
from r6_common import OUT, build_frame, feature_sets, temporal_masks

RNG = 42
rng = np.random.default_rng(RNG)
df, TYPE_COLS = build_frame()
SETS = feature_sets(TYPE_COLS)
FS = SETS["G4_full_plus_velocity"]
y = df.isFraud.values.astype(np.int8)
step = df.step.values
tr, va, te = temporal_masks(step)
X = df[FS].to_numpy(np.float32)
res = {"env": {"python": platform.python_version(), "xgboost": xgb.__version__,
               "machine": platform.machine(), "cpu": platform.processor(),
               "logical_cpus": os.cpu_count()}}

# ---- models
models = {}
b = xgb.Booster(); b.load_model(os.path.join(OUT, "xgb_temporal.json"))
n_t = b.num_boosted_rounds()
best_t = int(b.attr("best_iteration") or 7)
models["T8_temporal_best_iteration"] = (b, (0, best_t + 1), best_t + 1)
models["T58_temporal_all_stored"] = (b, (0, 0), n_t)
b2 = xgb.Booster(); b2.load_model(os.path.join(OUT, "xgb_random.json"))
models["R56_random_all_stored"] = (b2, (0, 0), b2.num_boosted_rounds())
print("loaded boosters", n_t, b2.num_boosted_rounds(), flush=True)

t0 = time.time()
spw = float((y[tr] == 0).sum() / y[tr].sum())
clf600 = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
                           tree_method="hist", n_jobs=-1, random_state=RNG)
clf600.fit(X[tr], y[tr], verbose=False)
models["X600_forced_600_rounds"] = (clf600.get_booster(), (0, 0), 600)
res["x600_train_seconds"] = time.time() - t0
print("trained 600-round model", flush=True)

rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=RNG)
t0 = time.time(); rf.fit(X[tr], y[tr]); res["rf_train_seconds"] = time.time() - t0
print("trained RF", flush=True)

sample_idx = rng.choice(np.flatnonzero(te), 20000, replace=False)
S = X[sample_idx]


def pct(a):
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean()),
            "n_ge_1ms": int((a >= 1.0).sum()), "n": int(len(a))}


def time_single(fn, n=3000, warm=500):
    lat = []
    for i in range(n + warm):
        row = S[i:i + 1]
        t0 = time.perf_counter(); fn(row); dt = (time.perf_counter() - t0) * 1000
        if i >= warm:
            lat.append(dt)
    return np.array(lat)


def time_batch(fn, reps=150, warm=20):
    lat = []
    for i in range(reps + warm):
        chunk = S[rng.integers(0, len(S), 1000)]
        t0 = time.perf_counter(); fn(chunk); dt = (time.perf_counter() - t0) * 1000
        if i >= warm:
            lat.append(dt)
    return np.array(lat)


def summarise_repeats(reps):
    keys = reps[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in reps])), "min": float(np.min([r[k] for r in reps])),
                "max": float(np.max([r[k] for r in reps]))} for k in keys}


res["single_event"] = {}; res["batch1000"] = {}
for name, (bst, it, ntrees) in models.items():
    for threads in ("default", 1):
        if threads == 1:
            bst.set_param({"nthread": 1})
        else:
            bst.set_param({"nthread": os.cpu_count()})
        fn = (lambda r, bst=bst, it=it: bst.inplace_predict(r, iteration_range=it))
        rep_s = [pct(time_single(fn)) for _ in range(5)]
        rep_b = [pct(time_batch(fn)) for _ in range(3)]
        for r in rep_b:
            r["rate_from_median_per_s"] = 1000 / (r["p50"] / 1000)
            r["rate_from_mean_per_s"] = 1000 / (r["mean"] / 1000)
        key = f"{name}|threads={threads}"
        res["single_event"][key] = {"trees": ntrees, "repeats": summarise_repeats(rep_s)}
        res["batch1000"][key] = {"trees": ntrees, "repeats": summarise_repeats(rep_b)}
        print(key, "single p50 %.4f p99 %.4f" % (res["single_event"][key]["repeats"]["p50"]["mean"],
                                                 res["single_event"][key]["repeats"]["p99"]["mean"]), flush=True)
    bst.set_param({"nthread": os.cpu_count()})

for threads in ("default", 1):
    rf.set_params(n_jobs=-1 if threads == "default" else 1)
    fn = lambda r: rf.predict_proba(r)
    rep_s = [pct(time_single(fn, n=500, warm=50)) for _ in range(3)]
    rep_b = [pct(time_batch(fn, reps=30, warm=5)) for _ in range(3)]
    for r in rep_b:
        r["rate_from_median_per_s"] = 1000 / (r["p50"] / 1000)
        r["rate_from_mean_per_s"] = 1000 / (r["mean"] / 1000)
    key = f"RF100_depth12|threads={threads}"
    res["single_event"][key] = {"trees": 100, "repeats": summarise_repeats(rep_s)}
    res["batch1000"][key] = {"trees": 100, "repeats": summarise_repeats(rep_b)}
    print(key, res["single_event"][key]["repeats"]["p50"], flush=True)

# ---- C. sustained runs (temporal best-iteration booster, default threads)
bst, it, nt = models["T8_temporal_best_iteration"]
bst.set_param({"nthread": os.cpu_count()})
dur = 30.0
n = 0; t_start = time.perf_counter(); lat = []
while time.perf_counter() - t_start < dur:
    row = S[n % 20000:n % 20000 + 1]
    t0 = time.perf_counter(); bst.inplace_predict(row, iteration_range=it); lat.append((time.perf_counter() - t0) * 1000)
    n += 1
el = time.perf_counter() - t_start
res["sustained_single_event_30s"] = {"events": n, "seconds": el, "events_per_s": n / el, **pct(np.array(lat))}
n = 0; t_start = time.perf_counter(); lat = []
while time.perf_counter() - t_start < dur:
    chunk = S[rng.integers(0, len(S), 1000)]
    t0 = time.perf_counter(); bst.inplace_predict(chunk, iteration_range=it); lat.append((time.perf_counter() - t0) * 1000)
    n += 1000
el = time.perf_counter() - t_start
res["sustained_batch1000_30s"] = {"events": n, "seconds": el, "events_per_s": n / el, **pct(np.array(lat))}
print("sustained", res["sustained_single_event_30s"]["events_per_s"], res["sustained_batch1000_30s"]["events_per_s"], flush=True)

# ---- D. end-to-end in-process with dictionary state
orig_codes = df.nameOrig.astype("category").cat.codes.values
dest_codes = df.nameDest.astype("category").cat.codes.values
amount = df.amount.values; ob = df.oldbalanceOrg.values; nb = df.newbalanceOrig.values
od = df.oldbalanceDest.values; nd = df.newbalanceDest.values
is_m = df.destIsMerchant.values; typ = df.type.values
type_order = TYPE_COLS  # alphabetical, as used for training
type_idx = {t.replace("type_", ""): k for k, t in enumerate(type_order)}
first_test = int(np.flatnonzero(te)[0])
state = IncrementalState()
t0 = time.time()
for i in range(first_test):
    state.update(step[i], amount[i], orig_codes[i], dest_codes[i])
res["state_warmup_seconds"] = time.time() - t0
run_idx = np.arange(first_test, first_test + 20000)
vec = np.empty((1, 23), dtype=np.float32)
t_feat, t_pred, t_total, scores = [], [], [], []
bst_e, it_e, _ = models["T8_temporal_best_iteration"]
for i in run_idx:
    tt0 = time.perf_counter()
    s, a, o, d = step[i], amount[i], orig_codes[i], dest_codes[i]
    oc, otsl, dc, dtsl, dmean, aod = state.features(s, a, o, d)
    err_o = nb[i] + a - ob[i]; err_d = od[i] + a - nd[i]
    zeroed = 1 if (ob[i] > 0 and nb[i] == 0) else 0
    vec[0, :18] = (a, s % 24, s // 24, ob[i], nb[i], od[i], nd[i], err_o, err_d, zeroed, is_m[i],
                   a / (ob[i] + 1.0), oc, otsl, dc, dtsl, dmean, aod)
    vec[0, 18:] = 0.0
    vec[0, 18 + type_idx[typ[i]]] = 1.0
    tt1 = time.perf_counter()
    sc = bst_e.inplace_predict(vec, iteration_range=it_e)
    tt2 = time.perf_counter()
    state.update(s, a, o, d)
    tt3 = time.perf_counter()
    t_feat.append((tt1 - tt0) * 1000); t_pred.append((tt2 - tt1) * 1000); t_total.append((tt3 - tt0) * 1000)
    scores.append(float(sc[0]))
scores = np.array(scores)
ref = bst_e.inplace_predict(X[run_idx], iteration_range=it_e)
res["end_to_end_in_process"] = {
    "events": int(len(run_idx)),
    "feature_build_ms": pct(np.array(t_feat)), "predict_ms": pct(np.array(t_pred)),
    "feature_plus_predict_plus_state_update_ms": pct(np.array(t_total)),
    "max_abs_score_difference_vs_matrix_path": float(np.max(np.abs(scores - ref))),
    "note": "dictionary stand-in for Redis, no network, no serialisation, single Python process",
}
print(res["end_to_end_in_process"], flush=True)

with open(os.path.join(OUT, "latency_v2.json"), "w") as f:
    json.dump(res, f, indent=1)
print("DONE")
