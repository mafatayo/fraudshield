"""
Latency budget breakdown chart for Section 4.2 (Objective One Results),
styled to match the other FraudShield figures. Uses only real numbers:
the allocated budget split from Table 3.3, and the measured model-scoring
latency (median and p99) from Section 3.5.2 / Table 4.4 -- the only stage
this study benchmarks in isolation. The other stages are explicitly
labelled "not benchmarked", matching Table 3.3's own honesty about what
was and wasn't measured on a single-node commodity setup.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.family": "serif", "font.size": 11})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2.3, 1]})

# ---- Left: allocated budget breakdown (Table 3.3, real allocation numbers) ----
stages = ["Ingestion &\nqueuing", "Feature computation\n& state access",
          "Model\nscoring", "Persistence &\nalert routing", "Headroom"]
ms = [100, 300, 50, 200, 350]
colors = ["#3a6ea5", "#3a6ea5", "#2f9e6e", "#6b6b6b", "#c98a2b"]
# Wide segments (>=150ms) get their label centred inside the bar; narrow
# segments get a leader line down to a label below the bar so text never
# has to squeeze into a bar narrower than the text itself.
WIDE_THRESHOLD = 150

left = 0
above_y, below_y = 0.7, -0.9
for s, v, c in zip(stages, ms, colors):
    ax1.barh(0, v, left=left, color=c, edgecolor="white", height=1.0, zorder=3)
    pct = v / 10
    label = f"{s}\n{v} ms ({pct:.0f}%)"
    center = left + v / 2
    if v >= WIDE_THRESHOLD:
        ax1.text(center, 0, label, ha="center", va="center", fontsize=9.5,
                 color="white", fontweight="bold", zorder=4)
    else:
        ax1.plot([center, center], [0.5, below_y + 0.35], color="#444444",
                 linewidth=0.9, zorder=2)
        ax1.text(center, below_y, label, ha="center", va="top", fontsize=9.2,
                 color="#222222", fontweight="bold")
    left += v

ax1.set_xlim(0, 1000)
ax1.set_ylim(-1.9, 1.2)
ax1.set_yticks([])
ax1.set_xlabel("Allocated latency budget (ms), 1,000 ms total end-to-end target", fontsize=10.5)
ax1.set_title("Design Allocation by Pipeline Stage  (Table 3.3)", fontsize=12, fontweight="bold", pad=14)
for spine in ["top", "right", "left"]:
    ax1.spines[spine].set_visible(False)

# ---- Right: measured vs allocated for the one benchmarked stage ----
import json, os
_L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "latency_v2.json")))["single_event"]
_T8 = _L["T8_temporal_best_iteration|threads=default"]["repeats"]; _T1 = _L["T8_temporal_best_iteration|threads=1"]["repeats"]
vals = [50, _T1["p50"]["mean"], _T8["p50"]["mean"], _T8["p99"]["mean"]]
labels = ["Allocated (50 ms)", "Median, 1 thread (%.2f ms)" % vals[1], "Median, all threads (%.2f ms)" % vals[2], "p99, all threads (%.2f ms)" % vals[3]]
labels = [l.replace(", ", ",\n", 1).replace(" (", "\n(") for l in labels]
bar_colors = ["#e0e0e0", "#2f9e6e", "#2f9e6e", "#2f9e6e"]
bars = ax2.bar(labels, vals, color=bar_colors, edgecolor="#444444", linewidth=1.1, width=0.6)
ax2.set_yscale("log")
ax2.set_ylim(0.1, 100)
ax2.set_ylabel("Milliseconds (log scale)")
ax2.set_title("Model-Scoring Stage:\nMeasured vs Allocated", fontsize=12, fontweight="bold")
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:.2f} ms" if v < 10 else f"{v:g} ms", ha="center",
             fontsize=9.5, fontweight="bold")
ax2.text(0.5, -0.38, "One in-process call; no Redis or network.\nThe allocation is a design assumption.",
         transform=ax2.transAxes, ha="center", fontsize=8.6, color="#555555")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

note = mpatches.Patch(color="none",
                       label="Only scoring was benchmarked on its own; the other allocations are design assumptions (local-store reference pipeline: see Section 4.5.3).")
fig.legend(handles=[note], loc="lower center", frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("../figures/fig10_latency_budget.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig10_latency_budget.png")
