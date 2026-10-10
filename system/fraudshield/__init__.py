"""FraudShield reference implementation of the Chapter Three scoring pipeline.

A runnable, fault-injectable version of Algorithm 3.1 in which Kafka is replaced by a
partitioned append-only file log, and Redis and PostgreSQL by SQLite tables with the same
atomicity guarantees (insert-if-absent, an atomic apply-once state update, a keyed alert
queue). The substitution is stated in the thesis (Section 3.5.5); no network hop is measured.
"""
