"""All 6,362,620 events, one partition, no faults: throughput, latency percentiles and the full set of
invariants against the offline references. Writes results/system_full_stream.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, build_log, load_stream, supervise, verify  # noqa: E402

stream, offline = load_stream()
n = len(stream)
d, counts = build_log("FULL", stream, n)
lat = os.path.join(d, "lat.npy")
crashes, secs, _ = supervise(d, 0, None, (), lat_file=lat)
v = verify(d, stream, offline, n, expected_offsets=counts)
a = np.load(lat) * 1e3
v.update({"n_events": n, "wall_seconds": secs, "events_per_second": n / secs, "crashes": crashes,
          "latency_ms": {k: float(np.percentile(a, q)) for k, q in [("p50", 50), ("p95", 95), ("p99", 99), ("p999", 99.9)]} | {"max": float(a.max())}})
json.dump(v, open(os.path.join(ROOT, "results", "system_full_stream.json"), "w"), indent=1)
print(json.dumps({k: v[k] for k in ("pass", "decisions", "events_per_second", "latency_ms", "tp", "fp", "fn")}))
