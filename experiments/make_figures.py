"""Publication figures for the FraudShield manuscript (300 dpi, IEEE-friendly)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito (CVD-safe, validated)
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRAY = "#666666"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.bbox": "tight",
})

with open(os.path.join(OUT, "metrics.json")) as f:
    M = json.load(f)

# ------------------------------------------------ Fig 1: architecture diagram
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)

def box(x, y, w, h, label, sub="", fc="#eef3f8", ec="#0072B2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2 + (2.2 if sub else 0), label,
            ha="center", va="center", fontsize=8, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.8, sub, ha="center", va="center",
                fontsize=6.5, color="#333333")

def arrow(x1, y1, x2, y2, label="", style="-|>", color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, lw=1.0,
                                 linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, ha="center",
                fontsize=6.3, color="#333333", style="italic")

box(1, 26, 15, 12, "Transaction\nSources", "wallets · cards\ncrypto ramps", fc="#f5f5f5", ec=GRAY)
box(22, 26, 15, 12, "Apache Kafka", "transactions\ntopic")
box(43, 26, 18, 12, "Spark Streaming", "feature\ncomputation")
box(47, 6, 12, 10, "Redis", "behavioral state\nvelocity counters", fc="#fdf3e7", ec=C_ORANGE)
box(67, 26, 15, 12, "XGBoost\nScorer", "sub-ms\ninference", fc="#e9f5f0", ec=C_GREEN)
box(88, 32, 11, 9, "Alert\nQueue", "analyst review", fc="#fbeee7", ec=C_VERM)
box(88, 15, 11, 9, "PostgreSQL", "transactions\n& scores", fc="#f5f5f5", ec=GRAY)

arrow(16, 32, 22, 32)
arrow(37, 32, 43, 32)
arrow(61, 32, 67, 32)
ax.text(64, 40.5, "feature\nvector", ha="center", fontsize=6.3,
        color="#333333", style="italic")
arrow(50, 26, 51, 16, "update", color=C_ORANGE)
arrow(56, 16, 57, 26, "lookup", color=C_ORANGE)
arrow(82, 34, 88, 36, "fraud score")
arrow(82, 30, 88, 20)
plt.savefig(os.path.join(FIG, "fig1_architecture.png"))
plt.close()
print("fig1 done", flush=True)

# ------------------------------------------------ Fig 2: dataset composition
d = M["dataset"]
types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
tot = [d["type_counts"][t] for t in types]
fr = [d["fraud_by_type"][t] for t in types]
x = np.arange(len(types))
fig, ax = plt.subplots(figsize=(3.5, 2.4))
b1 = ax.bar(x - 0.21, tot, 0.4, color=C_BLUE, label="All transactions", zorder=3)
b2 = ax.bar(x + 0.21, [max(v, 0.5) for v in fr], 0.4, color=C_VERM, label="Fraudulent", zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.5, 5e6)
ax.set_xticks(x, [t.replace("_", "\n") for t in types], fontsize=7)
ax.set_ylabel("Transactions (log scale)")
for i, v in enumerate(fr):
    ax.text(x[i] + 0.21, max(v, 0.5) * 1.6, f"{v:,}", ha="center", fontsize=6, color=C_VERM)
ax.legend(fontsize=6.5, loc="upper right")
plt.savefig(os.path.join(FIG, "fig2_dataset.png"))
plt.close()
print("fig2 done", flush=True)

# ------------------------------------------------ Fig 3/4: PR + ROC curves
def curves_panel(tag, ax_pr, ax_roc, title):
    z = np.load(os.path.join(OUT, f"curves_{tag}.npz"))
    y = z["y"]
    specs = [("LR", z["lr_prob"], C_BLUE, ":"),
             ("RF", z["rf_prob"], C_ORANGE, "--"),
             ("XGBoost", z["xgb_prob"], C_GREEN, "-")]
    for name, prob, c, ls in specs:
        p, r, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax_pr.plot(r, p, color=c, ls=ls, label=f"{name} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax_roc.plot(fpr, tpr, color=c, ls=ls, label=f"{name} (AUC={auc:.4f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(title, fontsize=8.5)
    ax_pr.set_xlim(0, 1.02); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(fontsize=6, loc="lower left")
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(title, fontsize=8.5)
    ax_roc.set_xscale("log"); ax_roc.set_xlim(1e-6, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=6, loc="upper left")

fig_pr, axs_pr = plt.subplots(1, 2, figsize=(7.0, 2.7))
fig_roc, axs_roc = plt.subplots(1, 2, figsize=(7.0, 2.7))
curves_panel("random", axs_pr[0], axs_roc[0], "(a) Stratified random split")
curves_panel("temporal", axs_pr[1], axs_roc[1], "(b) Temporal split")
fig_pr.tight_layout(); fig_pr.savefig(os.path.join(FIG, "fig3_pr_curves.png")); plt.close(fig_pr)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(FIG, "fig4_roc_curves.png")); plt.close(fig_roc)
print("fig3/4 done", flush=True)

# ------------------------------------------------ Fig 5: feature importance
imp = M["xgb_gain_importance"]
names = list(imp.keys())[:15][::-1]
vals = [imp[n] for n in names]
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.barh(names, vals, color=C_BLUE, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Mean gain (log scale)")
ax.tick_params(axis="y", labelsize=7)
plt.savefig(os.path.join(FIG, "fig5_importance.png"))
plt.close()
print("fig5 done", flush=True)

# ------------------------------------------------ Fig 6: latency CDF
z = np.load(os.path.join(OUT, "latency.npz"))
lat = np.sort(z["single"])
cdf = np.arange(1, len(lat) + 1) / len(lat)
L = M["latency_ms"]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot(lat, cdf * 100, color=C_BLUE)
for q, v in [(50, L["single_p50"]), (95, L["single_p95"]), (99, L["single_p99"])]:
    ax.axvline(v, color=C_VERM, lw=0.7, ls="--")
    ax.text(v * 1.05, 8 + (0 if q == 50 else (10 if q == 95 else 20)),
            f"p{q}={v:.2f} ms", fontsize=6.5, color=C_VERM)
ax.set_xscale("log")
ax.set_xlabel("Single-transaction scoring latency (ms)")
ax.set_ylabel("Cumulative share (%)")
ax.set_ylim(0, 102)
plt.savefig(os.path.join(FIG, "fig6_latency.png"))
plt.close()
print("fig6 done", flush=True)

# ------------------------------------------------ Fig 7: ablation
abl_path = os.path.join(OUT, "ablation.json")
if os.path.exists(abl_path):
    with open(abl_path) as f:
        A = json.load(f)
    labels = {
        "G1_transaction_only": "G1\ntxn only",
        "G2_plus_raw_balances": "G2\n+balances",
        "G3_plus_consistency": "G3\n+consistency",
        "G4_full_plus_velocity": "G4 full\n+velocity",
        "G5_txn_plus_velocity_no_balance": "G5 txn\n+velocity",
    }
    keys = list(labels)
    pr = [A[k]["pr_auc"] for k in keys]
    f1 = [A[k]["f1"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x - 0.2, pr, 0.38, color=C_BLUE, label="PR-AUC", zorder=3)
    ax.bar(x + 0.2, f1, 0.38, color=C_GREEN, label="F1", zorder=3)
    for i in range(len(keys)):
        ax.text(x[i] - 0.2, pr[i] + 0.015, f"{pr[i]:.3f}", ha="center", fontsize=5.6)
        ax.text(x[i] + 0.2, f1[i] + 0.015, f"{f1[i]:.3f}", ha="center", fontsize=5.6)
    ax.set_xticks(x, [labels[k] for k in keys], fontsize=6.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (temporal test set)")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    plt.savefig(os.path.join(FIG, "fig7_ablation.png"))
    plt.close()
    print("fig7 done", flush=True)
else:
    print("ablation.json not found — fig7 skipped", flush=True)
print("ALL FIGURES DONE", flush=True)
