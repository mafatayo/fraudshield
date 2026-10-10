"""
Sequence diagram for Algorithm 3.1 (Idempotent Transaction Scoring Pipeline), Round-6 version.

Six lifelines (Kafka, Spark job, PostgreSQL, Redis, XGBoost scorer, alert queue). Order of operations:
decision lookup, state reads, features, score, insert-if-absent, idempotent state update,
state_applied marker, keyed alert, then the Kafka offset commit. On replay the stored score is reused,
the state update is skipped if the marker is set, and the keyed alert is repeated safely. Fonts are 9 pt or larger.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(figsize=(11.5, 9.6))
ax.set_xlim(0, 12.0)
ax.set_ylim(0.8, 13.4)
ax.axis("off")

actors = [
    ("Kafka", 1.0, "#eaf1f8", "#3a6ea5"),
    ("Spark job", 3.0, "#eaf1f8", "#3a6ea5"),
    ("PostgreSQL", 5.0, "#f2f2f2", "#6b6b6b"),
    ("Redis", 7.0, "#fdf1e0", "#c98a2b"),
    ("XGBoost\nscorer", 9.0, "#e8f5ef", "#2f9e6e"),
    ("Alert\nqueue", 10.9, "#fbeee7", "#d55e00"),
]
TOP, BOTTOM = 12.9, 1.6
for name, x, face, edge in actors:
    ax.add_patch(FancyBboxPatch((x - 0.8, TOP - 0.5), 1.6, 0.8, boxstyle="round,pad=0.05,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x, TOP - 0.1, name, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=4)
    ax.add_line(Line2D([x, x], [TOP - 0.5, BOTTOM], color="#999999", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1))
xk, xs, xp, xr, xx, xa = [a[1] for a in actors]


def msg(y, x1, x2, label, step=None, dashed=False, above=True):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                 color="#333333", linestyle="dashed" if dashed else "solid", zorder=2))
    pre = f"{step}: " if step else ""
    ax.text((x1 + x2) / 2, y + 0.12, pre + label, ha="center", va="bottom", fontsize=9.5, style="italic",
            bbox=dict(fc="white", ec="none", pad=0.6))


rows = [
    (12.0, xk, xs, "consume_next_event (offset not committed)", "1", False),
    (11.3, xs, xp, "get_decision(transaction_id)", "2", False),
    (10.75, xp, xs, "record, or none", None, True),
    (10.1, xs, xr, "get origin state", "4", False),
    (9.55, xr, xs, "state", None, True),
    (8.95, xs, xr, "get destination state", "5", False),
    (8.4, xr, xs, "state", None, True),
    (7.75, xs, xx, "predict_proba(features)", "6, 7", False),
    (7.2, xx, xs, "fraud score", None, True),
    (6.55, xs, xp, "insert decision if absent", "7", False),
    (5.85, xs, xr, "apply_once(origin, txn_id)  [Lua, atomic]", "8", False),
    (5.2, xs, xr, "apply_once(destination, txn_id)", "8", False),
    (4.55, xs, xp, "mark state_applied", "8", False),
    (3.9, xs, xa, "enqueue_alert(key = txn_id)  if score >= threshold", "9", False),
    (3.25, xs, xk, "commit offset", "10", False),
]
for y, x1, x2, label, step, dashed in rows:
    msg(y, x1, x2, label, step, dashed)
ax.text(0.05, 2.7,
        "Step 3 (replay): if step 2 returns a record, steps 4 to 7 are skipped and the stored score is reused.\n"
        "Step 8 runs only if the record does not yet show state_applied;\n"
        "the keyed alert (9) and the commit (10) are repeated safely.",
        fontsize=9.5, color="#333333", ha="left", va="top")
ax.text(0.05, 1.95,
        "Solid arrows: calls. Dashed arrows: return values. Crash anywhere before step 10 is repaired by replay.",
        fontsize=9.5, color="#555555", ha="left", va="top")
plt.tight_layout()
plt.savefig("../figures/fig8_sequence.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig8_sequence.png")
