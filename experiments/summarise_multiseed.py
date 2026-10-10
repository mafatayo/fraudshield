"""Summarise results/multiseed.json into results/multiseed_summary.json (means, ranges, pair counts)."""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
r = json.load(open(os.path.join(OUT, "multiseed.json")))
seeds = sorted(r["seeds"])
summ = {"seeds": seeds, "temporal": {}, "random": {}, "temporal_pairs": {}}
for proto in ("temporal", "random"):
    names = list(r["seeds"][seeds[0]][proto])
    for n in names:
        vals = [r["seeds"][s][proto][n] for s in seeds]
        d = {}
        for f in ("pr_auc", "roc_auc", "recall", "fn", "fp", "fn_at_zero_fp", "fn_mid", "recall_at_zero_fp",
                  "recall_at_fpr_1e-4", "trees_used", "n_features"):
            if f in vals[0]:
                a = np.array([v[f] for v in vals], float)
                d[f] = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": [float(x) for x in a]}
        summ[proto][n] = d
pairs = {}
for s in seeds:
    for k, v in r["seeds"][s].get("temporal_pairs", {}).items():
        pairs.setdefault(k, []).append({"seed": s, **v})
for k, lst in pairs.items():
    sig = [x for x in lst if x["p_exact"] < 0.05]
    summ["temporal_pairs"][k] = {
        "per_seed": lst,
        "n_seeds_p_lt_0.05": len(sig),
        "n_seeds_a_catches_more": sum(1 for x in lst if x["a_only"] > x["b_only"]),
        "n_seeds_b_catches_more": sum(1 for x in lst if x["b_only"] > x["a_only"]),
        "a_only_total": sum(x["a_only"] for x in lst), "b_only_total": sum(x["b_only"] for x in lst)}
json.dump(summ, open(os.path.join(OUT, "multiseed_summary.json"), "w"), indent=1)
for proto in ("temporal", "random"):
    print("==", proto)
    for n, d in summ[proto].items():
        print(f"{n:48s} PR {d['pr_auc']['mean']:.5f} [{d['pr_auc']['min']:.5f},{d['pr_auc']['max']:.5f}]  "
              f"FN {d['fn']['mean']:.1f} [{d['fn']['min']:.0f},{d['fn']['max']:.0f}]  "
              f"FN0FP {d['fn_at_zero_fp']['mean']:.1f} [{d['fn_at_zero_fp']['min']:.0f},{d['fn_at_zero_fp']['max']:.0f}]  "
              f"FP {d['fp']['mean']:.1f}  trees {d.get('trees_used',{}).get('mean','-')}  nf {d.get('n_features',{}).get('mean','-')}")
for k, v in summ["temporal_pairs"].items():
    print(k, "p<0.05 in", v["n_seeds_p_lt_0.05"], "of", len(seeds), "A>B", v["n_seeds_a_catches_more"], "B>A", v["n_seeds_b_catches_more"], "totals", v["a_only_total"], v["b_only_total"])
