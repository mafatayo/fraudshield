"""Crash injection. A crash is os._exit, so nothing after the point runs (no finally, no flush)."""
import os
import random

POINTS = ["after_score", "after_insert", "between_state", "after_state", "after_mark", "after_alert"]


class Crasher:
    def __init__(self):
        rate = float(os.environ.get("FS_CRASH_RATE", "0"))
        self.rate = rate
        self.rng = random.Random(int(os.environ.get("FS_CRASH_SEED", "0")))
        self.only = os.environ.get("FS_CRASH_POINT")
        self.flagged_only = os.environ.get("FS_CRASH_FLAGGED_ONLY") == "1"   # crash only on events the model flags
        self.flagged = False
        self.min_txn = int(os.environ.get("FS_CRASH_MIN_TXN", "0"))   # crash only on events from this identifier on
        self.txn = 0

    def at(self, point):
        if self.txn < self.min_txn:
            return
        if self.flagged_only and not self.flagged:
            return
        if self.rate and (self.only in (None, "", point)) and self.rng.random() < self.rate:
            log = os.environ.get("FS_CRASH_LOG")
            if log:
                with open(log, "a") as f:
                    f.write(point + "\n")
            os._exit(137)
