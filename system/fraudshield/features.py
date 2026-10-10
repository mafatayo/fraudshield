"""Online feature computation from one event and the prior state of its two accounts.

Mirrors Section 3.3 and r6_common.build_frame; experiments/run_parity_replay.py shows that the
logic equals the offline columns on all 6,362,620 events.
"""
import numpy as np

from .config import FEATURES, TYPES


def compute(ev, o_state, d_state):
    """o_state / d_state: (cnt, last_step, amt_sum) or None. Returns a float32 vector."""
    a = ev["amount"]
    oo, no, od, nd = ev["oldbalanceOrg"], ev["newbalanceOrig"], ev["oldbalanceDest"], ev["newbalanceDest"]
    step = ev["step"]
    oc, ol, _ = o_state if o_state else (0, 0, 0.0)
    dc, dl, ds = d_state if d_state else (0, 0, 0.0)
    mean = ds / dc if dc else 0.0
    v = np.zeros(len(FEATURES), dtype=np.float32)
    v[0] = a
    v[1] = step % 24
    v[2] = step // 24
    v[3], v[4], v[5], v[6] = oo, no, od, nd
    v[7] = no + a - oo
    v[8] = od + a - nd
    v[9] = 1.0 if (oo > 0 and no == 0) else 0.0
    v[10] = 1.0 if ev["dest"].startswith("M") else 0.0
    v[11] = a / (oo + 1.0)
    v[12] = oc
    v[13] = (step - ol) if oc else -1
    v[14] = dc
    v[15] = (step - dl) if dc else -1
    v[16] = mean
    v[17] = a / (mean + 1.0)
    v[18 + TYPES.index(ev["type"])] = 1.0
    return v
