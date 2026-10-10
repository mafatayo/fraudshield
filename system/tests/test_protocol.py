"""Unit tests for the idempotent protocol (Algorithm 3.1) on a small synthetic stream.

They need no dataset: the shipped model scores synthetic events. Run:  python -m pytest system/tests
"""
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from fraudshield import worker  # noqa: E402
from fraudshield.faults import POINTS  # noqa: E402
from fraudshield.scorer import Scorer  # noqa: E402
from fraudshield.stores import AlertQueue, DecisionStore, Offsets, StateStore, connect  # noqa: E402


class Crash(Exception):
    pass


class ScriptedCrasher:
    """Raises Crash the first time each (event, point) pair listed in `plan` is reached."""
    def __init__(self, plan):
        self.plan = set(plan)
        self.current = None

    def at(self, point):
        if (self.current, point) in self.plan:
            self.plan.discard((self.current, point))
            raise Crash(point)


class NoCrash:
    def at(self, point):
        pass


def synthetic(n, seed=1, n_accts=40):
    r = random.Random(seed)
    accts = [f"C{i}" for i in range(n_accts)] + [f"M{i}" for i in range(10)]
    out = []
    for t in range(n):
        o, d = r.choice(accts[:n_accts]), r.choice(accts)
        amt = r.choice([50.0, 900.0, 25000.0, 181.0])
        bal = r.choice([0.0, 181.0, 5000.0, 30000.0])
        out.append({"txn_id": t, "step": 1 + t // 8, "type": r.choice(["TRANSFER", "CASH_OUT", "PAYMENT"]),
                    "amount": amt, "oldbalanceOrg": bal, "newbalanceOrig": max(bal - amt, 0.0),
                    "oldbalanceDest": 0.0, "newbalanceDest": amt, "orig": o, "dest": d})
    return out


def fresh():
    c = connect(os.path.join(tempfile.mkdtemp(), "s.db"))
    return c, DecisionStore(c), StateStore(c), AlertQueue(c)


def snapshot(c):
    st = {r[0]: (r[1], r[2], round(r[3], 6)) for r in c.execute("SELECT acct,cnt,last_step,amt_sum FROM acct_state")}
    dec = {r[0]: r[1] for r in c.execute("SELECT txn_id,score FROM decisions")}
    al = {r[0] for r in c.execute("SELECT txn_id FROM alerts")}
    return st, dec, al


def run(events, crasher, deliveries=None, use_flag=True):
    c, dec, state, alerts = fresh()
    sc = Scorer()
    for ev in (deliveries or events):
        while True:                      # a crashed consumer is restarted and re-delivered the event
            crasher.current = ev["txn_id"]
            try:
                worker.process(ev, dec, state, alerts, sc, crasher, use_flag)
                break
            except Crash:
                continue
    return c


def test_crash_at_every_point_equals_clean_run():
    ev = synthetic(300)
    clean = snapshot(run(ev, NoCrash()))
    plan = [(e["txn_id"], p) for e in ev[::7] for p in POINTS]
    assert snapshot(run(ev, ScriptedCrasher(plan))) == clean


def test_every_event_has_one_decision_and_applied_state():
    ev = synthetic(200)
    c = run(ev, ScriptedCrasher([(e["txn_id"], "after_insert") for e in ev[::3]]))
    assert c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 200
    assert c.execute("SELECT COUNT(*) FROM decisions WHERE state_applied=0").fetchone()[0] == 0


def test_late_redelivery_does_not_double_count_but_needs_the_flag():
    ev = synthetic(800, n_accts=4)   # few accounts, so each sees far more events than the recent list holds
    deliveries = ev + [ev[3], ev[50], ev[120]]            # redelivered long after the recent-id list moved on
    clean = snapshot(run(ev, NoCrash()))
    assert snapshot(run(ev, NoCrash(), deliveries)) == clean
    assert snapshot(run(ev, NoCrash(), deliveries, use_flag=False))[0] != clean[0]   # negative control


def test_alert_queue_discards_repeated_key():
    c, dec, state, alerts = fresh()
    assert alerts.enqueue(7, 0.99) is True
    assert alerts.enqueue(7, 0.99) is False
    assert c.execute("SELECT n FROM alert_attempts").fetchone()[0] == 2


def test_insert_if_absent_has_one_winner():
    c, dec, state, alerts = fresh()
    assert dec.insert_if_absent(1, 0.2, False, 0.7, "v", b"") is True
    assert dec.insert_if_absent(1, 0.9, True, 0.7, "v", b"") is False
    assert dec.get(1)["score"] == 0.2


def test_apply_once_is_idempotent_within_the_recent_list():
    c, dec, state, alerts = fresh()
    assert state.apply_once("o:C1", 5, 1, 100.0) is True
    assert state.apply_once("o:C1", 5, 1, 100.0) is False
    assert state.get("o:C1") == (1, 1, 100.0)


def test_offset_commit_resumes_where_it_stopped():
    c = connect(os.path.join(tempfile.mkdtemp(), "s.db"))
    o = Offsets(c, "g")
    assert o.get(0) == 0
    o.commit(0, 41)
    assert Offsets(c, "g").get(0) == 41
