"""
Machine learning paradigm taxonomy diagram for Section 2.2.3, styled to
match fig1_architecture.png / fig8_sequence.png's palette and typography.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.size": 11})

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)
ax.axis("off")


def box(x, y, w, h, text, face, edge, fontsize=11, bold=True):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.05,rounding_size=0.09",
                        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             fontweight="bold" if bold else "normal", zorder=4)


root_x, root_y = 5.0, 4.9
box(root_x, root_y, 3.6, 0.75, "Machine Learning Paradigms", "#eaf1f8", "#3a6ea5", 12)

children = [
    ("Supervised", "labelled\nnormal + fraud", 1.0, "#e8f5ef", "#2f9e6e", True),
    ("Unsupervised", "no labels;\nstatistical rarity", 3.2, "#f2f2f2", "#6b6b6b", False),
    ("Semi-\nsupervised", "normal-only\nlabels", 5.0, "#f2f2f2", "#6b6b6b", False),
    ("Reinforcement", "reward-driven\napprove/decline", 6.9, "#f2f2f2", "#6b6b6b", False),
    ("Self-\nsupervised", "pretext task on\nunlabelled data", 8.85, "#f2f2f2", "#6b6b6b", False),
]

branch_y = 3.55
leaf_y = 2.35
sub_y = 1.55

# connector from root down to a horizontal spine, then to each branch
ax.add_line(Line2D([root_x, root_x], [root_y - 0.375, branch_y + 0.55], color="#555555", linewidth=1.3, zorder=1))
ax.add_line(Line2D([children[0][2], children[-1][2]], [branch_y + 0.55, branch_y + 0.55], color="#555555", linewidth=1.3, zorder=1))

for name, sub, x, face, edge, adopted in children:
    ax.add_line(Line2D([x, x], [branch_y + 0.55, branch_y + 0.375], color="#555555", linewidth=1.3, zorder=1))
    box(x, branch_y, 1.55, 0.75, name, face, edge, 10.5)
    ax.add_line(Line2D([x, x], [branch_y - 0.375, leaf_y + 0.32], color="#999999", linewidth=1.0,
                        linestyle=(0, (3, 2)), zorder=1))
    ax.text(x, leaf_y, sub, ha="center", va="center", fontsize=9, style="italic", color="#333333")
    if adopted:
        ax.text(x, sub_y, "adopted by\nthis study", ha="center", va="center", fontsize=8.8,
                 color="#1f6b45", fontweight="bold")
        ax.add_patch(FancyBboxPatch((x - 0.78, sub_y - 0.28), 1.56, 0.56,
                                     boxstyle="round,pad=0.03,rounding_size=0.06",
                                     linewidth=1.1, edgecolor="#2f9e6e", facecolor="none",
                                     linestyle="dashed", zorder=2))

ax.text(0.05, 0.35,
        "Dashed connectors: the labelled-data requirement distinguishing each paradigm, per Table 2.3.",
        fontsize=8.6, color="#555555", ha="left")

plt.tight_layout()
plt.savefig("../figures/fig9_ml_taxonomy.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig9_ml_taxonomy.png")
