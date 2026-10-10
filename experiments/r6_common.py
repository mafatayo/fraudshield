"""
Shared data preparation for the Round-6 analyses (run_multiseed.py,
run_consistency_stats.py, run_parity_replay.py, run_latency_v2.py).

build_frame() reproduces, line for line, the feature construction used by
run_experiments.py, run_ablation.py and run_causal_comparison.py (stable sort
by step, then the same pandas groupby/cumcount/shift definitions), so the
seed-42 runs in run_multiseed.py can be checked against the existing
results/ablation.json and results/causal_comparison.json.

It adds two things the earlier scripts did not have:
  * wouldDrainOrig, a feature that is knowable BEFORE funds move
    (amount >= oldbalanceOrg with a positive pre-transaction balance);
  * named feature sets, including the pre-authorisation sets, which use
    only fields available before the transaction posts.
"""
import os

import numpy as np
import pandas as pd

RNG = 42
DATA = os.environ.get(
    "PAYSIM_CSV",
    r"C:\Users\VICTUS2025\.cache\kagglehub\datasets\ealaxi\paysim1\versions\2\PS_20174392719_1491204439457_log.csv",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def build_frame(path=DATA):
    df = pd.read_csv(path)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df["hour"] = df.step % 24
    df["day"] = df.step // 24
    df["errBalOrig"] = df.newbalanceOrig + df.amount - df.oldbalanceOrg
    df["errBalDest"] = df.oldbalanceDest + df.amount - df.newbalanceDest
    df["origZeroedOut"] = ((df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)).astype(np.int8)
    df["destIsMerchant"] = df.nameDest.str.startswith("M").astype(np.int8)
    df["amountToBalanceRatio"] = df.amount / (df.oldbalanceOrg + 1.0)
    # pre-authorisation drain indicator: uses only the pre-transaction balance
    df["wouldDrainOrig"] = ((df.oldbalanceOrg > 0) & (df.amount >= df.oldbalanceOrg)).astype(np.int8)

    g_orig = df.groupby("nameOrig", sort=False)
    df["origPriorTxnCount"] = g_orig.cumcount()
    df["origTimeSinceLast"] = (df.step - g_orig["step"].shift(1)).fillna(-1)
    g_dest = df.groupby("nameDest", sort=False)
    df["destPriorTxnCount"] = g_dest.cumcount()
    df["destTimeSinceLast"] = (df.step - g_dest["step"].shift(1)).fillna(-1)
    prior_amt_sum = g_dest["amount"].cumsum() - df.amount
    df["destPriorAmtMean"] = (prior_amt_sum / df.destPriorTxnCount.replace(0, np.nan)).fillna(0.0)
    df["amountOverDestMean"] = df.amount / (df.destPriorAmtMean + 1.0)

    # non-causal ("leaky") counterparts, as in run_causal_comparison.py
    dest_full_count = g_dest["amount"].transform("count") - 1
    dest_full_mean = (g_dest["amount"].transform("sum") - df.amount) / dest_full_count.replace(0, np.nan)
    df["destPriorTxnCount_leaky"] = dest_full_count.fillna(0)
    df["destPriorAmtMean_leaky"] = dest_full_mean.fillna(0.0)
    df["amountOverDestMean_leaky"] = df.amount / (df.destPriorAmtMean_leaky + 1.0)

    type_d = pd.get_dummies(df.type, prefix="type", dtype=np.int8)
    df = pd.concat([df, type_d], axis=1)
    return df, list(type_d.columns)


TXN = ["amount", "hour", "day"]
BAL = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CONS = ["errBalOrig", "errBalDest", "origZeroedOut", "destIsMerchant", "amountToBalanceRatio"]
VEL = ["origPriorTxnCount", "origTimeSinceLast", "destPriorTxnCount",
       "destTimeSinceLast", "destPriorAmtMean", "amountOverDestMean"]
# fields that are only known once the transaction has posted to the ledger
POST_ONLY = ["newbalanceOrig", "newbalanceDest", "errBalOrig", "errBalDest", "origZeroedOut"]


def feature_sets(type_cols):
    base_full = TXN + BAL + CONS + VEL
    sets = {
        "G1_transaction_only": TXN + type_cols,
        "G2_plus_raw_balances": TXN + BAL + type_cols,
        "G3_plus_consistency": TXN + BAL + CONS + type_cols,
        "G4_full_plus_velocity": base_full + type_cols,
        "G5_txn_plus_velocity_no_balance": TXN + VEL + type_cols,
        "G4_noncausal_leaky": ["amount", "hour", "day", "oldbalanceOrg", "newbalanceOrig",
                               "oldbalanceDest", "newbalanceDest", "errBalOrig", "errBalDest",
                               "origZeroedOut", "destIsMerchant", "amountToBalanceRatio",
                               "origPriorTxnCount", "origTimeSinceLast",
                               "destPriorTxnCount_leaky", "destTimeSinceLast",
                               "destPriorAmtMean_leaky", "amountOverDestMean_leaky"] + type_cols,
        # ---- pre-authorisation sets: nothing that is known only after posting
        # PRE: pre-transaction balances, drain indicators, destination type, velocity
        "PRE_full": TXN + ["oldbalanceOrg", "oldbalanceDest", "amountToBalanceRatio",
                           "wouldDrainOrig", "destIsMerchant"] + VEL + type_cols,
        # PRE_noVel: the same without the streaming-state (velocity) family
        "PRE_noVel": TXN + ["oldbalanceOrg", "oldbalanceDest", "amountToBalanceRatio",
                            "wouldDrainOrig", "destIsMerchant"] + type_cols,
        # PRE_raw: raw pre-transaction fields only, no engineered ratio or drain flag
        "PRE_raw_velocity": TXN + ["oldbalanceOrg", "oldbalanceDest", "destIsMerchant"] + VEL + type_cols,
        # full post-posting set without the simulation-clock index `day`
        "G4_noDay": [c for c in base_full if c != "day"] + type_cols,
    }
    return sets


def temporal_masks(step):
    q60, q70 = np.quantile(step, [0.60, 0.70])
    return step <= q60, (step > q60) & (step <= q70), step > q70
