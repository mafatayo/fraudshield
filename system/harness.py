"""Build a log, supervise worker processes through crashes, and verify the invariants."""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "system"))
from fraudshield.config import RUN_DIR  # noqa: E402
from fraudshield.log import LogWriter  # noqa: E402
from fraudshield.scorer import Scorer  # noqa: E402
from fraudshield.stores import connect  # noqa: E402

PY = sys.executable


def load_stream():
    return pd.read_pickle(os.path.join(RUN_DIR, "stream.pkl")), np.load(os.path.join(RUN_DIR, "offline_scores.npy"))


def build_log(name, stream, n_events, n_parts=1, dup_frac=0.0, dup_max_gap=5000, seed=0, hot_dups=0, start=0):
    """Write n_events events from row `start` on (txn_id = row index). dup_frac of them are delivered a second time
    later in the stream, up to dup_max_gap positions after the original."""
    d = os.path.join(RUN_DIR, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "log"))
    rng = np.random.default_rng(seed)
    order = np.arange(n_events, dtype=np.float64)
    ids = np.arange(n_events)
    if dup_frac:
        k = int(n_events * dup_frac)
        pick = rng.choice(n_events, size=k, replace=False)
        later = pick + rng.integers(1, dup_max_gap, size=k) + 0.5
        order = np.concatenate([order, later])
        ids = np.concatenate([ids, pick])
        ids = ids[np.argsort(order, kind="stable")]
    if hot_dups:
        # redeliver, after the whole prefix, events whose destination account is busy (40 or more events)
        # so that many later events have pushed the identifier out of that account's recent list
        dest = stream.nameDest.iloc[start:start + n_events]
        busy = dest.map(dest.value_counts()) >= 40
        cand = np.flatnonzero(busy.to_numpy() & (np.arange(n_events) < n_events // 2))
        pick = rng.choice(cand, size=min(hot_dups, len(cand)), replace=False)
        ids = np.concatenate([ids, np.sort(pick)])
    w = LogWriter(os.path.join(d, "log"), n_parts)
    ids = ids + start
    cols = [stream[c].to_numpy() for c in ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                                           "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest"]]
    for i in ids:
        w.append(int(i), int(cols[0][i]), cols[1][i], float(cols[2][i]), float(cols[3][i]), float(cols[4][i]),
                 float(cols[5][i]), float(cols[6][i]), cols[7][i], cols[8][i])
    w.close()
    connect(os.path.join(d, "store.db")).close()
    return d, w.count


def load_checkpoint(run_dir, stream, start):
    """Restore the per-account state as it stands after the first `start` events (a state checkpoint), so a run
    can begin in the middle of the stream with realistic, busy accounts."""
    st = replay_reference_state(stream.iloc[:start])
    c = connect(os.path.join(run_dir, "store.db"))
    c.execute("BEGIN")
    c.executemany("INSERT INTO acct_state(acct,cnt,last_step,amt_sum,recent) VALUES(?,?,?,?,',')",
                  [(k, v[0], v[1], v[2]) for k, v in st.items()])
    c.execute("COMMIT")
    c.close()
    return len(st)


def supervise(run_dir, partition, env_extra=None, args=(), group="g1", max_restarts=5000, lat_file=""):
    """Run a worker; restart it whenever it exits non-zero. Returns (crashes, seconds, crashes by point)."""
    crashes, t0 = 0, time.time()
    env = dict(os.environ, **(env_extra or {}))
    crashlog = os.path.join(run_dir, f"crashes_p{partition}_{group}.log")
    env["FS_CRASH_LOG"] = crashlog
    while True:
        env["FS_CRASH_SEED"] = str(int(env.get("FS_CRASH_SEED", "0")) + 1)
        cmd = [PY, "-m", "fraudshield.worker", run_dir, str(partition), "--group", group, *args]
        if lat_file:
            cmd += ["--lat-file", lat_file]
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "system"), env=env)
        if r.returncode == 0:
            break
        if r.returncode != 137:
            raise RuntimeError(f"worker failed with {r.returncode}")
        crashes += 1
        if crashes > max_restarts:
            raise RuntimeError("too many restarts")
    by_point = {}
    if os.path.exists(crashlog):
        for line in open(crashlog):
            by_point[line.strip()] = by_point.get(line.strip(), 0) + 1
    return crashes, time.time() - t0, by_point


def replay_reference_state(s):
    st = {}
    for step, amt, o, d in zip(s.step.values, s.amount.values, s.nameOrig.values, s.nameDest.values):
        for key in ("o:" + o, "d:" + d):
            e = st.get(key)
            if e is None:
                st[key] = [1, int(step), float(amt)]
            else:
                e[0] += 1
                e[1] = int(step)
                e[2] += float(amt)
    return st


def verify(run_dir, stream, offline, n_events, scorer=None, baseline_db=None, expected_offsets=None, start=0):
    """Check the thesis invariants against the store and the offline references."""
    scorer = scorer or Scorer()
    c = connect(os.path.join(run_dir, "store.db"))
    out = {}
    rows = c.execute("SELECT txn_id, score, flagged, state_applied FROM decisions ORDER BY txn_id").fetchall()
    ids = np.array([r[0] for r in rows])
    sc = np.array([r[1] for r in rows])
    out["decisions"] = len(rows)
    out["missing_decisions"] = int(n_events - len(set(ids.tolist()) & set(range(start, start + n_events))))
    out["state_applied_false"] = int(sum(1 for r in rows if not r[3]))
    ref = offline[ids].astype(np.float64)
    out["score_mismatches_vs_offline"] = int((np.abs(sc - ref) > 2e-7).sum())   # float32 rounding of the leaf sum
    out["max_abs_score_diff"] = float(np.abs(sc - ref).max()) if len(sc) else 0.0
    flagged_ref = set(ids[ref >= scorer.threshold].tolist())
    alerts = {r[0] for r in c.execute("SELECT txn_id FROM alerts")}
    out["alerts_delivered"] = len(alerts)
    out["alerts_expected"] = len(flagged_ref)
    out["alert_set_equal"] = alerts == flagged_ref
    out["alert_attempts"] = c.execute("SELECT n FROM alert_attempts").fetchone()[0]
    out["duplicate_alerts_suppressed"] = out["alert_attempts"] - out["alerts_delivered"]
    rows_log = c.execute("SELECT COUNT(*), COUNT(DISTINCT txn_id) FROM alert_log").fetchone()
    out["unkeyed_alert_rows"] = rows_log[0]
    out["unkeyed_alert_duplicates"] = rows_log[0] - rows_log[1]
    s = stream.iloc[start:start + n_events]
    st = replay_reference_state(stream.iloc[:start + n_events])
    got = {r[0]: (r[1], r[2], r[3]) for r in c.execute("SELECT acct,cnt,last_step,amt_sum FROM acct_state")}
    bad = 0
    for k, v in st.items():
        g = got.get(k)
        if g is None or g[0] != v[0] or g[1] != v[1] or abs(g[2] - v[2]) > 1e-6 * max(1.0, abs(v[2])):
            bad += 1
    out["accounts"] = len(st)
    out["state_accounts_in_store"] = len(got)
    out["state_mismatches"] = bad + abs(len(got) - len(st))
    out["offsets"] = {int(r[0]): int(r[1]) for r in c.execute("SELECT part,next_offset FROM offsets")}
    out["fraud_in_window"] = int(s.isFraud.sum())
    pred = sc >= scorer.threshold
    yy = stream.isFraud.values[ids]
    out["tp"] = int((pred & (yy == 1)).sum())
    out["fp"] = int((pred & (yy == 0)).sum())
    out["fn"] = int((~pred & (yy == 1)).sum())
    if baseline_db and os.path.exists(baseline_db):
        b = connect(baseline_db)
        base = dict(b.execute("SELECT txn_id, score FROM decisions"))
        b.close()
        common = [(t, x) for t, x in zip(ids.tolist(), sc.tolist()) if t in base]
        out["compared_with_fault_free_run"] = len(common)
        out["bit_differences_vs_fault_free_run"] = int(sum(1 for t, x in common if base[t] != x))
    offsets_ok = expected_offsets is None or [out["offsets"].get(i) for i in range(len(expected_offsets))] == list(expected_offsets)
    out["offsets_equal_log_length"] = bool(offsets_ok)
    bits_ok = out.get("bit_differences_vs_fault_free_run", 0) == 0
    out["pass"] = bool(offsets_ok and bits_ok and out["unkeyed_alert_duplicates"] == 0 and out["missing_decisions"] == 0 and out["decisions"] == n_events and out["state_applied_false"] == 0
                       and out["score_mismatches_vs_offline"] == 0 and out["alert_set_equal"]
                       and out["state_mismatches"] == 0)
    c.close()
    return out
