"""
Sequence diagram for Algorithm 3.1 (Synchronous Transaction Scoring Pipeline),
styled to match fig1_architecture.png's palette and typography.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.size": 11})

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 9.6)
ax.axis("off")

actors = [
    ("Kafka",       1.0,  "#eaf1f8", "#3a6ea5"),
    ("Spark\nStreaming", 3.2, "#eaf1f8", "#3a6ea5"),
    ("Redis",       5.4,  "#fdf1e0", "#c98a2b"),
    ("XGBoost\nScorer",  7.4, "#e8f5ef", "#2f9e6e"),
    ("Alert Queue\n/ PostgreSQL", 9.6, "#f2f2f2", "#6b6b6b"),
]

TOP = 9.0
BOTTOM = 0.6

# lifeline boxes + dashed lifelines
for name, x, face, edge in actors:
    box = FancyBboxPatch((x - 0.85, TOP - 0.45), 1.7, 0.7,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(box)
    ax.text(x, TOP - 0.1, name, ha="center", va="center", fontsize=10.5,
             fontweight="bold", zorder=4)
    ax.add_line(Line2D([x, x], [TOP - 0.45, BOTTOM], color="#999999",
                        linestyle=(0, (4, 3)), linewidth=1.1, zorder=1))

xk, xs, xr, xx, xa = [a[1] for a in actors]


def msg(y, x_from, x_to, label, dashed=False, step=None):
    style = "->" if not dashed else "-|>"
    arrow = FancyArrowPatch((x_from, y), (x_to, y),
                             arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.4, color="#333333",
                             linestyle="dashed" if dashed else "solid", zorder=2)
    ax.add_patch(arrow)
    mid = (x_from + x_to) / 2
    prefix = f"{step}: " if step is not None else ""
    ax.text(mid, y + 0.14, f"{prefix}{label}", ha="center", va="bottom",
             fontsize=9.3, style="italic")


rows = [
    (8.1, xk, xs, "consume_next_event(kafka_topic)", False, 1),
    (7.5, xs, xr, "redis_get(origin_account_id)", False, 2),
    (6.9, xr, xs, "prior behavioural state", True, None),
    (6.2, xs, xs, "compute_features(e, state)  [Alg. 3.2]", False, 3),
    (5.5, xs, xx, "predict_proba(x)", False, 4),
    (4.9, xx, xs, "fraud score s", True, None),
    (4.2, xs, xa, "route_to_review_or_block(e, s)  if s ≥ τ", False, "5–7"),
    (3.4, xs, xr, "redis_update(origin_account_id, e)", False, 8),
    (2.6, xs, xa, "postgres_persist(e, x, s)", False, 9),
]
for r in rows:
    y, xf, xt, label, dashed, step = r
    if xf == xt:
        # self-call loop on Spark's lifeline
        ax.add_patch(FancyArrowPatch((xf, y + 0.05), (xf + 0.9, y - 0.25),
                                      connectionstyle="arc3,rad=-1.4",
                                      arrowstyle="-|>", mutation_scale=13,
                                      linewidth=1.3, color="#333333", zorder=2))
        ax.text(xf + 1.0, y - 0.1, f"{step}: {label}", ha="left", va="center",
                 fontsize=9.3, style="italic")
    else:
        msg(y, xf, xt, label, dashed, step)

ax.text(0.05, BOTTOM - 0.25,
        "Solid arrows: synchronous calls in Algorithm 3.1's numbered order.  Dashed arrows: return values.",
        fontsize=8.6, color="#555555", ha="left")

plt.tight_layout()
plt.savefig("../figures/fig8_sequence.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig8_sequence.png")
