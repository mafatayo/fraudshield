# FraudShield reference implementation

A runnable version of Algorithm 3.1 of the thesis (the idempotent transaction-scoring pipeline), small enough to
crash on purpose.

| Thesis design | Reference implementation |
|---|---|
| Kafka topic, partitioned by origin account | Append-only file log, one file per partition (`fraudshield/log.py`) |
| Consumer-group offset commit | A row in the `offsets` table, written last |
| PostgreSQL decision record, insert-if-absent | SQLite `decisions` table, `INSERT OR IGNORE` |
| Redis hashes and the apply-once Lua script | SQLite `acct_state` table and one atomic upsert statement |
| Keyed alert queue | `alerts` table with the transaction identifier as primary key |
| Spark Streaming job | One worker process per partition, one event at a time (`fraudshield/worker.py`) |
| XGBoost scorer | The 8 trees of the temporal model, read from `model/xgb_deploy.json` and walked in Python |

The substitution keeps the logic (order of steps, atomic units, crash windows, deduplication keys) and drops the
infrastructure. It says nothing about network latency, replication, or the failure of a broker, cache or database.
A crash here is the death of the worker process (`os._exit`), not of the machine.

## Run

```bash
pip install -r requirements.txt
python -m pytest system/tests                 # protocol tests on a synthetic stream; no dataset needed

# the experiments need the PaySim CSV (Kaggle: ealaxi/paysim1); set PAYSIM_CSV or use the kagglehub cache
python system/train_model.py                  # trains the temporal XGBoost model, writes system/model/
python system/prepare_reference.py            # caches the event stream and the offline scores
python system/check_scorer_parity.py          # tree-walk scorer against XGBoost on all 6,362,620 events
python system/run_fault_injection.py          # E1 to E6: crashes, duplicates, negative control
python system/run_scaling.py                  # 1, 2 and 4 consumer processes
python system/run_full_stream.py              # all 6,362,620 events, with the full set of checks
```

Results are written to `results/system_*.json`.

## What the checks assert

After every run the store is compared with independent references:

1. every transaction identifier has exactly one decision, and its `state_applied` marker is set;
2. each score is within 2e-7 of the offline model and bit-identical to the fault-free run;
3. the final per-account state equals a dictionary replay of the distinct events;
4. the set of alerts equals the set of flagged events, with no event alerted twice;
5. the committed offset equals the length of the log.

Experiment `E4hn` ignores the `state_applied` marker on late duplicates of events on busy accounts and fails check 3
(44 account states wrong), while `E4h`, which keeps the marker, passes. That pair shows the checks can detect a
double-applied update. `E4n`, the same control on sparse accounts, passes, because the recent-identifier list is rarely overrun.
