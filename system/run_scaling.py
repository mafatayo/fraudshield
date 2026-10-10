"""Throughput with 1, 2 and 4 consumer processes (one partition each, keyed by origin account).
Destination-account state is shared between partitions, so with more than one consumer the order in
which two partitions touch the same destination account is not the stream order (Section 3.2.1).
This script measures both the throughput and how often that changes a score or a flag.
Writes results/system_scaling.json."""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import PY, ROOT, build_log, load_stream  # noqa: E402
from fraudshield.scorer import Scorer  # noqa: E402
from fraudshield.stores import connect  # noqa: E402

N = int(os.environ.get("FS_N", "300000"))
stream, offline = load_stream()
sc = Scorer()
out = {"n_events": N}
for P in (1, 2, 4):
    name = f"S{P}"
    d, counts = build_log(name, stream, N, n_parts=P)
    t0 = time.time()
    procs = [subprocess.Popen([PY, "-m", "fraudshield.worker", d, str(p)], cwd=os.path.join(ROOT, "system")) for p in range(P)]
    for pr in procs:
        pr.wait()
    secs = time.time() - t0
    c = connect(os.path.join(d, "store.db"))
    rows = c.execute("SELECT txn_id, score FROM decisions ORDER BY txn_id").fetchall()
    ids = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows])
    ref = offline[ids].astype(np.float64)
    y = stream.isFraud.values[ids]
    pred = s >= sc.threshold
    refpred = ref >= sc.threshold
    out[f"P{P}"] = {
        "partitions": P, "records_per_partition": counts, "wall_seconds": secs, "events_per_second": N / secs,
        "decisions": len(rows), "score_differs_from_stream_order": int((np.abs(s - ref) > 2e-7).sum()),
        "flag_differs_from_stream_order": int((pred != refpred).sum()),
        "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()), "fn": int((~pred & (y == 1)).sum()),
        "fraud_in_prefix": int(y.sum()),
    }
    print(name, json.dumps(out[f"P{P}"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "system_scaling.json"), "w"), indent=1)
