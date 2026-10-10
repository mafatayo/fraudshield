"""
Incremental (per-event) implementation of the six stateful feature columns,
using plain Python dictionaries as a stand-in for the Redis hashes of the
design in Chapter Three. It is written independently of the pandas
groupby/cumcount/shift code in r6_common.build_frame so that run_parity_replay.py
can test whether the two implementations agree.
"""


class IncrementalState:
    __slots__ = ("o_cnt", "o_last", "d_cnt", "d_last", "d_sum")

    def __init__(self):
        self.o_cnt = {}
        self.o_last = {}
        self.d_cnt = {}
        self.d_last = {}
        self.d_sum = {}

    def features(self, step, amount, orig, dest):
        """Read-only: the six features for an event, from prior state only."""
        oc = self.o_cnt.get(orig, 0)
        o_tsl = (step - self.o_last[orig]) if oc else -1
        dc = self.d_cnt.get(dest, 0)
        d_tsl = (step - self.d_last[dest]) if dc else -1
        mean = (self.d_sum[dest] / dc) if dc else 0.0
        return (oc, o_tsl, dc, d_tsl, mean, amount / (mean + 1.0))

    def update(self, step, amount, orig, dest):
        """Apply the event to the state (done after scoring)."""
        self.o_cnt[orig] = self.o_cnt.get(orig, 0) + 1
        self.o_last[orig] = step
        self.d_cnt[dest] = self.d_cnt.get(dest, 0) + 1
        self.d_last[dest] = step
        self.d_sum[dest] = self.d_sum.get(dest, 0.0) + amount
