"""Fault-injection and correctness experiments for the reference implementation.
Writes results/system_fault_injection.json (resumable per experiment).

E1   no-fault run: parity with the offline model, throughput, latency
E2   random crashes at all six points
E3_* targeted crashes, one crash point at a time
E4   5% redelivered events, up to 5000 positions late (replay path, state_applied flag)
E4n  negative control: E4 with the state_applied flag disabled
E5   E4 plus random crashes
E4h  2,000 events on busy destination accounts redelivered after the prefix
E4hn the same with the marker ignored (negative control)
E7   random crashes with the offset committed every 100 events (multi-event replay)
E8   out-of-sample window with busy accounts: state checkpoint at the start of the test window, duplicates, crashes, batched commits
E9   two consumers race on one partition
E10  negative control: offset committed before processing (must fail)
E11  negative control: alert queue without a key (must fail)
E6_* crashes only on flagged events, one crash point at a time (alert path)
usage: python run_fault_injection.py [experiment ...]   (default: those not yet in the result file)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, build_log, load_checkpoint, load_stream, supervise, verify  # noqa: E402
from fraudshield.faults import POINTS  # noqa: E402

OUTP = os.path.join(ROOT, "results", "system_fault_injection.json")
RES = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
only = set(sys.argv[1:])
stream, offline = load_stream()


def save():
    json.dump(RES, open(OUTP, "w"), indent=1)


def go(key, n, **kw):
    if only and key not in only:
        return
    if key in RES and not only:
        print("skip", key)
        return
    crash_env = kw.pop("env", None)
    args = kw.pop("args", ())
    start = kw.get("start", 0)
    expected_fail = kw.pop("expected_fail", False)
    d, counts = build_log(key, stream, n, **kw)
    if start:
        load_checkpoint(d, stream, start)
    lat = os.path.join(d, "lat.npy") if key == "E1" else ""
    crashes, secs, by_point = supervise(d, 0, crash_env, args, lat_file=lat)
    v = verify(d, stream, offline, n, baseline_db=None if key == "E1" else os.path.join(os.path.dirname(d), "E1", "store.db"), expected_offsets=counts, start=start)
    v.update({"start_row": start, "expected_failure": expected_fail, "n_events": n, "log_records": counts, "crashes": crashes, "crashes_by_point": by_point,
              "wall_seconds": secs, "events_per_second": sum(counts) / secs})
    if lat and os.path.exists(lat):
        a = np.load(lat) * 1e3
        v["latency_ms"] = {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                           "p99": float(np.percentile(a, 99)), "p999": float(np.percentile(a, 99.9)),
                           "max": float(a.max())}
    RES[key] = v
    save()
    keys = ("pass", "crashes", "decisions", "score_mismatches_vs_offline", "state_mismatches", "duplicate_alerts_suppressed", "bit_differences_vs_fault_free_run")
    print(key, json.dumps({k: v.get(k) for k in keys}), flush=True)


N = int(os.environ.get("FS_N", "100000"))
go("E1", N)
go("E2", N, env={"FS_CRASH_RATE": "0.0003"})
for p in POINTS:
    go("E3_" + p, 30000, env={"FS_CRASH_RATE": "0.002", "FS_CRASH_POINT": p})
go("E4", N, dup_frac=0.05)
go("E4n", N, dup_frac=0.05, args=("--no-applied-flag",))
go("E5", N, dup_frac=0.05, env={"FS_CRASH_RATE": "0.0003"})
# E6: crashes only on events the model flags, so that the alert path and the state of fraudulent events are hit
for p in ("after_insert", "between_state", "after_mark", "after_alert"):
    go("E6_" + p, 30000, env={"FS_CRASH_RATE": "0.5", "FS_CRASH_POINT": p, "FS_CRASH_FLAGGED_ONLY": "1"})
# E4h: late redelivery of events on busy destination accounts, after the whole prefix; E4hn ignores the marker
go("E4h", N, hot_dups=2000)
go("E4hn", N, hot_dups=2000, args=("--no-applied-flag",))
# E7: the offset is committed every 100 events, so a crash replays up to 100 events (as a micro-batch replays a batch)
go("E7", N, env={"FS_CRASH_RATE": "0.0003"}, args=("--commit-every", "100"))

# ---- out-of-sample window with busy accounts: state is restored from a checkpoint taken at the start of the temporal
# test window, then 100,000 events are processed with duplicates, crashes and batched offset commits
import numpy as _np
_step = stream.step.values
_start = int(_np.argmax(_step > _np.quantile(_step, 0.70)))      # first row of the temporal test window
go("E8", N, start=_start, dup_frac=0.05, env={"FS_CRASH_RATE": "0.0003"}, args=("--commit-every", "100"))

# ---- negative controls for other mechanisms (these must FAIL)
# E10: the offset is committed before the event is processed, so a crash loses the event
go("E10", N, env={"FS_CRASH_RATE": "0.0003"}, args=("--commit-before",), expected_fail=True)
# E11: the alert queue does not deduplicate, so a crash after the alert sends it twice
go("E11", 30000, env={"FS_CRASH_RATE": "0.5", "FS_CRASH_POINT": "after_alert", "FS_CRASH_FLAGGED_ONLY": "1"}, args=("--no-alert-key",), expected_fail=True)


# ---- E9: two consumers race on the same partition (same group, no faults)
def race():
    import threading
    key = "E9"
    if (only and key not in only) or (key in RES and not only):
        return
    d, counts = build_log(key, stream, N)
    out = {}

    def one(i):
        out[i] = supervise(d, 0, None, ())

    ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    v = verify(d, stream, offline, N, baseline_db=os.path.join(os.path.dirname(d), "E1", "store.db"), expected_offsets=counts)
    v.update({"n_events": N, "log_records": counts, "consumers": 2, "expected_failure": False,
              "wall_seconds": max(o[1] for o in out.values()), "crashes": 0, "crashes_by_point": {}})
    v["events_per_second"] = N / v["wall_seconds"]
    RES[key] = v
    save()
    print(key, json.dumps({k: v.get(k) for k in ("pass", "decisions", "score_mismatches_vs_offline", "state_mismatches", "missing_decisions", "bit_differences_vs_fault_free_run")}), flush=True)


race()
