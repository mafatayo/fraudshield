"""
Round-6 figure regeneration (fonts of at least 8 pt, British spelling, no overlapping labels).
Writes figures/fig1, fig2, fig3, fig4, fig5, fig7 and results/importance_temporal.json.

  fig1  architecture, with the decision return path and the label offsets fixed
  fig2  dataset composition, with CASH_OUT style labels
  fig3  precision-recall curves with a zoomed inset on the top-right corner
  fig4  ROC curves on a log false-positive axis
  fig5  gain importance of the TEMPORAL headline model (8 trees in use), bars coloured
        by whether the feature is known before the transaction posts
  fig7  ablation: five-seed means with min-max bars, post-posting and pre-authorisation sets
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4, "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.bbox": "tight",
})
M = json.load(open(os.path.join(OUT, "metrics.json")))

# ------------------------------------------------------------------ fig1 architecture
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(-2, 62)


def box(x, y, w, h, label, sub="", fc="#eef3f8", ec=C_BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2 + (2.6 if sub else 0), label, ha="center", va="center", fontsize=9, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 3.2, sub, ha="center", va="center", fontsize=8, color="#333333")


def arrow(x1, y1, x2, y2, color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1, linestyle=ls))


def lab(x, y, s, color="#333333", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=8, color=color, style="italic",
            bbox=dict(fc="white", ec="none", pad=1.2))


box(1, 36, 16, 15, "Transaction\nsources", "wallets, agents", fc="#f5f5f5", ec="#666666")
box(24, 36, 15, 15, "Kafka", "transaction\nlog")
box(46, 36, 18, 15, "Spark job", "features, state\nreads and updates")
box(46, 8, 18, 14, "Redis", "account state", fc="#fdf3e7", ec=C_ORANGE)
box(71, 36, 14, 15, "XGBoost\nscorer", "stateless", fc="#e9f5f0", ec=C_GREEN)
box(90, 44, 13, 12, "Alert\nqueue", "analyst review", fc="#fbeee7", ec=C_VERM)
box(90, 10, 13, 14, "PostgreSQL", "decision\nrecords", fc="#f5f5f5", ec="#666666")
arrow(17.5, 43.5, 23.5, 43.5)
arrow(39.5, 43.5, 45.5, 43.5)
arrow(64.5, 46.5, 70.5, 46.5); lab(67.6, 50.4, "features")
arrow(70.5, 41, 64.5, 41, ls="--"); lab(67.6, 37.3, "score")
arrow(52, 35.5, 52, 22.8, color=C_ORANGE); lab(50.2, 27, "update", color=C_ORANGE, ha="right")
arrow(58, 22.8, 58, 35.5, color=C_ORANGE); lab(60.3, 27, "read", color=C_ORANGE, ha="left")
# alert path: above the scorer
ax.plot([55, 55, 96.5], [51.8, 58.5, 58.5], color="#333333", lw=1.1); arrow(96.5, 58.5, 96.5, 56.7)
lab(76, 58.5, "alert if score is at or above the threshold")
# persistence path: below the scorer
ax.plot([78, 78, 96.5], [35.4, 3, 3], color="#333333", lw=1.1); arrow(96.5, 3, 96.5, 9.3)
lab(86, 3, "insert decision record")
# decision return path to the payment system, routed below the first row
ax.plot([47, 47, 9], [35.2, 29.5, 29.5], color=C_GREEN, lw=1.3, ls="--")
arrow(9, 29.5, 9, 35.2, color=C_GREEN)
lab(28, 26.5, "approve, hold or decline returned to the payment system", color=C_GREEN)
plt.savefig(os.path.join(FIG, "fig1_architecture.png")); plt.close()
print("fig1", flush=True)

# ------------------------------------------------------------------ fig2 dataset
d = M["dataset"]
types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
tot = [d["type_counts"][t] for t in types]; fr = [d["fraud_by_type"][t] for t in types]
x = np.arange(len(types))
fig, ax = plt.subplots(figsize=(3.8, 2.6))
ax.bar(x - 0.21, tot, 0.4, color=C_BLUE, label="All transactions", zorder=3)
ax.bar(x + 0.21, [max(v, 0.5) for v in fr], 0.4, color=C_VERM, label="Fraudulent", zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.5, 5e7)
ax.set_xticks(x, types, fontsize=7, rotation=20)
ax.set_ylabel("Transactions (log scale)")
for i, v in enumerate(fr):
    ax.text(x[i] + 0.21, max(v, 0.5) * 1.8, f"{v:,}", ha="center", fontsize=7, color=C_VERM)
ax.legend(fontsize=7, loc="upper right")
plt.savefig(os.path.join(FIG, "fig2_dataset.png")); plt.close()
print("fig2", flush=True)

# ------------------------------------------------------------------ fig3 / fig4
def panel(tag, ax_pr, ax_roc, title):
    z = np.load(os.path.join(OUT, f"curves_{tag}.npz")); y = z["y"]
    specs = [("LR", z["lr_prob"], C_BLUE, ":", 1.4, None), ("RF", z["rf_prob"], C_ORANGE, "--", 2.6, None),
             ("XGBoost", z["xgb_prob"], C_GREEN, "-", 1.1, "o")]
    inset = inset_axes(ax_pr, width="42%", height="42%", loc="lower left", bbox_to_anchor=(0.1, 0.1, 1, 1),
                       bbox_transform=ax_pr.transAxes)
    for name, prob, c, ls, lw, mk in specs:
        p, r, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        kw = dict(color=c, ls=ls, lw=lw)
        ax_pr.plot(r, p, label=f"{name} (AP={ap:.4f})", **kw)
        inset.plot(r, p, **kw)
        fpr, tpr, _ = roc_curve(y, prob); auc = roc_auc_score(y, prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.5f})", **kw)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.set_title(title, fontsize=9)
    ax_pr.set_xlim(0, 1.02); ax_pr.set_ylim(0, 1.02); ax_pr.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(0.93, 0.86))
    inset.set_xlim(0.98, 1.0005); inset.set_ylim(0.97, 1.003); inset.tick_params(labelsize=7)
    inset.set_title("zoom: top-right corner", fontsize=7.5)
    ax_roc.set_xlabel("False positive rate (log scale)"); ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(title, fontsize=9); ax_roc.set_xscale("log"); ax_roc.set_xlim(1e-6, 1)
    ax_roc.set_ylim(0.4, 1.01); ax_roc.legend(fontsize=7.5, loc="lower right")

f1_, a1 = plt.subplots(1, 2, figsize=(7.4, 3.2)); f2_, a2 = plt.subplots(1, 2, figsize=(7.4, 3.2))
panel("random", a1[0], a2[0], "(a) Stratified random split")
panel("temporal", a1[1], a2[1], "(b) Temporal split")
f1_.tight_layout(); f1_.savefig(os.path.join(FIG, "fig3_pr_curves.png")); plt.close(f1_)
f2_.tight_layout(); f2_.savefig(os.path.join(FIG, "fig4_roc_curves.png")); plt.close(f2_)
print("fig3/4", flush=True)

# ------------------------------------------------------------------ fig5 importance (temporal headline model)
b = xgb.Booster(); b.load_model(os.path.join(OUT, "xgb_temporal.json"))
best = int(b.attr("best_iteration") or 7)
sub = b[0:best + 1]
names = (["amount", "hour", "day", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
          "errBalOrig", "errBalDest", "origZeroedOut", "destIsMerchant", "amountToBalanceRatio",
          "origPriorTxnCount", "origTimeSinceLast", "destPriorTxnCount", "destTimeSinceLast",
          "destPriorAmtMean", "amountOverDestMean", "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
          "type_PAYMENT", "type_TRANSFER"])
imp_raw = sub.get_score(importance_type="gain")
imp = {(names[int(k[1:])] if re.fullmatch(r"f\d+", k) else k): v for k, v in imp_raw.items()}
imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
tot = sum(imp.values())
post_only = {"newbalanceOrig", "newbalanceDest", "errBalOrig", "errBalDest", "origZeroedOut"}
json.dump({"model": "xgb_temporal.json, first %d trees (best iteration + 1)" % (best + 1),
           "mean_gain": imp, "share_of_summed_mean_gain": {k: v / tot for k, v in imp.items()},
           "post_posting_only_share": sum(v for k, v in imp.items() if k in post_only) / tot},
          open(os.path.join(OUT, "importance_temporal.json"), "w"), indent=1)
top = list(imp)[:14][::-1]
fig, ax = plt.subplots(figsize=(3.9, 3.5))
cols = [C_VERM if n in post_only else C_BLUE for n in top]
ax.barh(top, [imp[n] for n in top], color=cols, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("Mean gain per split (log scale)"); ax.tick_params(axis="y", labelsize=7.5)
ax.barh([], [], color=C_VERM, label="known only after posting"); ax.barh([], [], color=C_BLUE, label="known before posting")
ax.legend(fontsize=7, loc="lower right")
plt.savefig(os.path.join(FIG, "fig5_importance.png")); plt.close()
print("fig5", imp, flush=True)

# ------------------------------------------------------------------ fig7 ablation (five seeds)
S = json.load(open(os.path.join(OUT, "multiseed_summary.json")))["temporal"]
order = [("xgb:G1_transaction_only", "G1\ntxn only"), ("xgb:G2_plus_raw_balances", "G2\n+balances"),
         ("xgb:G3_plus_consistency", "G3\n+consistency"), ("xgb:G4_full_plus_velocity", "G4 full\n+velocity"),
         ("xgb:G5_txn_plus_velocity_no_balance", "G5 txn\n+velocity"), ("xgb:PRE_full", "PRE_full\npre-auth"),
         ("xgb:PRE_noVel", "PRE_noVel\npre-auth"), ("xgb:PRE_raw_velocity", "PRE_raw\npre-auth")]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
xs = np.arange(len(order))
means = [S[k]["pr_auc"]["mean"] for k, _ in order]
lo = [S[k]["pr_auc"]["mean"] - S[k]["pr_auc"]["min"] for k, _ in order]
hi = [S[k]["pr_auc"]["max"] - S[k]["pr_auc"]["mean"] for k, _ in order]
colors = [C_BLUE] * 5 + [C_GREEN] * 3
ax.bar(xs, means, color=colors, zorder=3, yerr=[lo, hi], capsize=3, error_kw=dict(lw=0.9))
for i, m in enumerate(means):
    ax.text(xs[i], m + 0.045, f"{m:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(xs, [l for _, l in order], fontsize=7.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("PR-AUC, temporal test window")
ax.text(0.0, 1.08, "blue: post-posting feature sets    green: pre-authorisation sets    bars: mean of five seeds, whiskers: min to max",
        fontsize=7, transform=ax.transData)
plt.savefig(os.path.join(FIG, "fig7_ablation.png")); plt.close()
print("fig7 done", flush=True)
