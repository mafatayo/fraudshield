"""Response time under offered load.

The other runs replay the log as fast as the worker can read it, so the time per event is service time with no
queueing. Here events are offered at a fixed rate (event k is due at k / rate seconds after the start) and the
response time of each event is the time from its due time to the commit of its offset. If the worker falls behind,
events queue and the response time grows. The deadline is the one-second end-to-end budget of Table 3.3.
Writes results/system_paced.json.

usage: python run_paced.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, build_log, load_stream, supervise, verify  # noqa: E402

RATES = [350, 1000, 2500, 4000, 6000, 8000]   # 350 is the annual average rate used for sizing in Section 3.2.7
SECONDS = 30
stream, offline = load_stream()
out = {"seconds_per_rate": SECONDS, "deadline_ms": 1000.0, "rates": {}}
for r in RATES:
    n = int(r * SECONDS)
    d, counts = build_log(f"PACED{r}", stream, n)
    lat = os.path.join(d, "lat.npy")
    _, secs, _ = supervise(d, 0, None, ("--rate", str(r)), lat_file=lat)
    a = np.load(lat) * 1e3                              # columns: service time, response time (ms)
    svc, resp = a[:, 0], a[:, 1]
    v = verify(d, stream, offline, n, expected_offsets=counts)
    out["rates"][str(r)] = {
        "offered_per_second": r, "events": n, "wall_seconds": secs, "achieved_per_second": n / secs,
        "service_ms": {k: float(np.percentile(svc, q)) for k, q in [("p50", 50), ("p99", 99)]},
        "response_ms": {**{k: float(np.percentile(resp, q)) for k, q in [("p50", 50), ("p95", 95), ("p99", 99), ("p999", 99.9)]}, "max": float(resp.max())},
        "events_over_100ms": int((resp > 100).sum()), "events_over_deadline": int((resp > 1000).sum()),
        "invariants_pass": v["pass"],
    }
    print(r, json.dumps(out["rates"][str(r)]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "system_paced.json"), "w"), indent=1)
