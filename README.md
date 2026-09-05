# FraudShield

A real-time, streaming machine learning system for fraud detection in high-volume financial transaction streams — built and evaluated on the full 6.36M-transaction PaySim dataset, with a causally-engineered feature scheme, a dual-protocol (random + temporal) evaluation, a feature-group ablation, and a measured latency budget.

This is the research codebase behind my MSc thesis at Lead City University and a manuscript prepared for *Bridges* (FASCON 2026), under double-blind review — the manuscript itself isn't included here for that reason. A companion write-up on the QA methodology used to structure this project's own tooling decisions is on [Zenodo](https://doi.org/10.5281/zenodo.22341782).

## The problem

Rule-based fraud detection — fixed thresholds, cooldowns, blacklists — catches almost nothing. On this dataset, a straightforward rule baseline catches **16 of 8,213** real fraud cases (recall 0.19%). Machine learning does far better, but most published fraud-detection studies on this dataset evaluate under a random train/test split, which quietly lets the model see transactions "from the future" relative to some test rows — a real deployed system never can. This project asks a narrower, more honest question: **how well does a causally-correct, streaming-architected model actually perform when it can only ever see the past — and how much of a near-perfect result is real signal versus an artifact of the benchmark itself?**

## Architecture

![Streaming architecture](figures/fig1_architecture.png)

Every transaction is published to **Kafka**, feature-computed on the fly by **Spark Streaming** against per-account behavioral state held in **Redis**, scored by a trained **XGBoost** model, and persisted to **PostgreSQL** for the audit trail — all against an explicit one-second, end-to-end latency budget.

## Results

| Detector | Split | PR-AUC | Precision | Recall | Missed frauds |
|---|---|---|---|---|---|
| Rule baseline | — | — | 1.000 | 0.19% | 8,197 / 8,213 |
| Logistic Regression | random | 0.565 | 0.625 | 0.548 | — |
| Random Forest | random | 0.998 | 1.000 | 0.996 | 7 / 1,643 |
| **XGBoost** | random | **0.9986** | 1.000 | 0.997 | 5 / 1,643 |
| **XGBoost** | **temporal** | **0.9997** | 1.000 | 0.995 | 22 / 4,570 |

- **Latency**: 0.21ms median, 0.81ms p99 per-transaction scoring (single node, i5-13420H)
- **Throughput**: ~1.85M transactions/second, batched
- **Training time**: XGBoost 14.0s vs. Random Forest 139.5s (temporal protocol) — the gap that makes frequent retraining against concept drift affordable

### Feature-group ablation — the honest part

| Feature group | Features | PR-AUC |
|---|---|---|
| Transaction attributes only | 8 | 0.294 |
| + raw balances | 12 | 0.969 |
| + balance-consistency checks | 17 | 0.9996 |
| **Full set (+ velocity)** | **23** | **0.9995** |
| Transaction + velocity, *no* balance data | 14 | 0.409 |

Near-perfect accuracy on this dataset is largely explained by **balance-consistency features** — PaySim's fraud simulation doesn't correctly update the mule account's balance, so a simple accounting-identity check nearly gives the answer away. That's the simulator being simple, not the model cheating, and it's said openly rather than left for a reviewer to discover. The last row is the actual novel contribution: in the realistic counterfactual where balance data is unreliable (card-not-present, cross-platform), **velocity features** — the ones that require the Kafka/Redis streaming architecture to exist at all — are the difference between PR-AUC 0.29 and 0.41, and they cut missed frauds from 32 to 3 on top of the full feature set.

## Repository structure

```
experiments/
  run_experiments.py       # LR / RF / XGBoost, random + temporal splits, latency benchmark
  run_ablation.py          # five nested feature-group ablation (G1-G5)
  make_figures.py          # publication figures from results/
  make_sequence_diagram.py # Fig 8 — Algorithm 1 as a sequence diagram
  make_ml_taxonomy.py      # Fig 9 — ML paradigm taxonomy
  make_latency_budget_chart.py  # Fig 10 — latency budget vs. measured
results/
  metrics.json, ablation.json      # every number in the tables above, raw
  curves_random.npz, curves_temporal.npz, latency.npz
figures/
  fig1_architecture.png … fig10_latency_budget.png  (300 dpi)
```

## Reproducing this

```bash
pip install -r requirements.txt
```

The dataset is [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle). Download it and point `DATA` at the top of `experiments/run_experiments.py` and `experiments/run_ablation.py` to your local copy — e.g. via `kagglehub.dataset_download("ealaxi/paysim1")`. Then:

```bash
python experiments/run_experiments.py   # ~5 min, writes results/metrics.json + figures 2-6
python experiments/run_ablation.py      # ~2 min, writes results/ablation.json + figure 7
python experiments/make_figures.py      # regenerate figures from saved results only
```

## Causal feature scheme

23 features across three families, every one constructed so it can only use information that existed strictly before the transaction being scored:

- **Transaction attributes** — amount, hour, day, type
- **Balance-consistency** — accounting-identity checks (does the sender's new balance equal old balance minus amount?)
- **Velocity** — per-account behavioral state (prior transaction count, time since last activity) — computed with a cumulative count/shift keyed to transaction order, never a naive groupby aggregate, to eliminate look-ahead leakage

## Limitations, stated directly

- PaySim's fraud pattern is stationary — this proves the model doesn't cheat by looking at the future, but can't prove it holds up against an adversary that adapts over time (real concept drift).
- Latency figures are single-node, in-process, single-transaction — the fully assembled pipeline under concurrent load has not been benchmarked.
- The F1-maximized decision threshold is a defensible default, not a claim about a production cost function.

## Citation

```
Fatayo, M. A. "FraudShield: A Real-Time Machine Learning Framework for Fraud
Detection in High-Volume Financial Transaction Streams." Manuscript prepared
for Bridges (FASCON 2026), Lead City University, Ibadan.
```

## License

MIT — see [LICENSE](LICENSE).
