"""One consumer: Algorithm 3.1, with named crash points.

usage: python -m fraudshield.worker RUN_DIR PARTITION [--group G] [--no-applied-flag] [--stop-after N]
"""
import argparse
import os
import sys
import time

from . import features
from .faults import Crasher
from .log import LogReader
from .scorer import Scorer
from .stores import AlertQueue, DecisionStore, Offsets, StateStore, connect


def process(ev, dec, state, alerts, scorer, crash, use_applied_flag=True):
    """Handle one event. Returns (score, was_replay)."""
    txn = ev["txn_id"]
    crash.txn = txn
    rec = dec.get(txn)                                                   # step 2
    replay = rec is not None
    if rec is None:                                                      # steps 4 to 7
        x = features.compute(ev, state.get("o:" + ev["orig"]), state.get("d:" + ev["dest"]))
        s = scorer.score(x)
        flagged = s >= scorer.threshold
        crash.flagged = flagged
        crash.at("after_score")
        if not dec.insert_if_absent(txn, s, flagged, scorer.threshold, scorer.version, x.tobytes()):
            rec = dec.get(txn)                                           # lost a race: use the winner
            replay = True
        crash.at("after_insert")
    if rec is not None:
        s = rec["score"]
        flagged = rec["flagged"]          # the decision as it was stored, not re-derived from the current threshold
        crash.flagged = flagged
    applied = rec["state_applied"] if (rec is not None and use_applied_flag) else False
    if not applied:                                                      # step 8
        state.apply_once("o:" + ev["orig"], txn, ev["step"], ev["amount"])
        crash.at("between_state")
        state.apply_once("d:" + ev["dest"], txn, ev["step"], ev["amount"])
        crash.at("after_state")
        dec.mark_applied(txn)
        crash.at("after_mark")
    if flagged:                                                          # step 9
        alerts.enqueue(txn, s)
    crash.at("after_alert")
    return s, replay


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("partition", type=int)
    ap.add_argument("--group", default="g1")
    ap.add_argument("--no-applied-flag", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--lat-file", default="")
    ap.add_argument("--rate", type=float, default=0.0, help="paced arrivals: events offered per second (0: as fast as possible)")
    ap.add_argument("--commit-before", action="store_true", help="negative control: commit the offset before processing")
    ap.add_argument("--no-alert-key", action="store_true", help="negative control: alert queue without deduplication")
    ap.add_argument("--commit-every", type=int, default=1, help="commit the offset every K events (K=1: after each event)")
    a = ap.parse_args(argv)

    conn = connect(os.path.join(a.run_dir, "store.db"))
    dec, state, alerts, offs = DecisionStore(conn), StateStore(conn), AlertQueue(conn, not a.no_alert_key), Offsets(conn, a.group)
    scorer, crash = Scorer(), Crasher()
    reader = LogReader(os.path.join(a.run_dir, "log"), a.partition)
    lat, resp = [], []
    n_done = 0
    t_start = time.perf_counter()
    while True:
        off = offs.get(a.partition)
        batch = reader.read_from(off, 512)
        if not batch:
            break
        for ev in batch:
            if a.rate:                                    # paced arrivals: the event is offered at k / rate seconds
                due = t_start + n_done / a.rate
                while time.perf_counter() < due:
                    if due - time.perf_counter() > 0.002:
                        time.sleep(0.001)
            else:
                due = None
            t0 = time.perf_counter()
            if a.commit_before:
                offs.commit(a.partition, off + 1)         # negative control: an event is lost if the worker dies now
            process(ev, dec, state, alerts, scorer, crash, not a.no_applied_flag)           # steps 2 to 9
            off += 1
            if off % a.commit_every == 0:
                offs.commit(a.partition, off)                                               # step 10
            t1 = time.perf_counter()
            if a.lat_file:
                lat.append(t1 - t0)
                resp.append(t1 - due if due is not None else t1 - t0)
            n_done += 1
            if a.stop_after and n_done >= a.stop_after:
                break
        offs.commit(a.partition, off)                  # end of the batch read: commit what was processed
        if a.stop_after and n_done >= a.stop_after:
            break
    if a.lat_file:
        import numpy as np
        arr = np.array(lat, dtype=np.float64)
        if a.rate:
            arr = np.column_stack([arr, np.array(resp, dtype=np.float64)])   # service time, response time
        np.save(a.lat_file, arr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
