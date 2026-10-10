"""Decision store, state store, alert queue and offset store (SQLite stand-ins).

Each method is one SQLite transaction, so it has the atomicity that Algorithm 3.1 assumes
of the real component it replaces:
  DecisionStore.insert_if_absent   <->  INSERT ... ON CONFLICT DO NOTHING in PostgreSQL
  StateStore.apply_once            <->  the Lua script in Redis (check list, update, record)
  AlertQueue.enqueue               <->  a queue that discards a repeated deduplication key
  Offsets.commit                   <->  the Kafka consumer-group offset commit
"""
import sqlite3

from .config import RECENT_CHARS

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions(
  txn_id INTEGER PRIMARY KEY, score REAL NOT NULL, flagged INTEGER NOT NULL,
  threshold REAL NOT NULL, model_version TEXT NOT NULL, features BLOB, state_applied INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS acct_state(
  acct TEXT PRIMARY KEY, cnt INTEGER NOT NULL, last_step INTEGER NOT NULL, amt_sum REAL NOT NULL, recent TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts(txn_id INTEGER PRIMARY KEY, score REAL NOT NULL);
CREATE TABLE IF NOT EXISTS alert_attempts(n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS alert_log(id INTEGER PRIMARY KEY AUTOINCREMENT, txn_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS offsets(grp TEXT NOT NULL, part INTEGER NOT NULL, next_offset INTEGER NOT NULL, PRIMARY KEY(grp, part));
"""


def connect(path):
    c = sqlite3.connect(path, timeout=60, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=OFF")
    c.executescript(SCHEMA)
    if c.execute("SELECT COUNT(*) FROM alert_attempts").fetchone()[0] == 0:
        c.execute("INSERT INTO alert_attempts VALUES(0)")
    return c


class DecisionStore:
    def __init__(self, c):
        self.c = c

    def get(self, txn_id):
        r = self.c.execute("SELECT score, flagged, state_applied FROM decisions WHERE txn_id=?", (txn_id,)).fetchone()
        return None if r is None else {"score": r[0], "flagged": bool(r[1]), "state_applied": bool(r[2])}

    def insert_if_absent(self, txn_id, score, flagged, threshold, version, features):
        cur = self.c.execute("INSERT OR IGNORE INTO decisions(txn_id,score,flagged,threshold,model_version,features) VALUES(?,?,?,?,?,?)",
                             (txn_id, score, int(flagged), threshold, version, features))
        return cur.rowcount == 1

    def mark_applied(self, txn_id):
        self.c.execute("UPDATE decisions SET state_applied=1 WHERE txn_id=?", (txn_id,))


class StateStore:
    def __init__(self, c):
        self.c = c

    def get(self, key):
        r = self.c.execute("SELECT cnt,last_step,amt_sum FROM acct_state WHERE acct=?", (key,)).fetchone()
        return r  # None for an account never seen

    def apply_once(self, key, txn_id, step, amount):
        """One atomic statement, like the Lua script: skip if txn_id is among the recently applied
        identifiers, otherwise update the counters (last step as a running maximum, so an update that arrives
        out of order cannot move it back) and append the identifier (list capped by length)."""
        t = "," + str(txn_id) + ","
        cur = self.c.execute(
            "INSERT INTO acct_state(acct,cnt,last_step,amt_sum,recent) VALUES(?,1,?,?,?) "
            "ON CONFLICT(acct) DO UPDATE SET cnt=cnt+1, last_step=max(last_step, excluded.last_step), "
            "amt_sum=amt_sum+excluded.amt_sum, recent=substr(recent||?, -?) "
            "WHERE instr(recent, ?)=0",
            (key, step, amount, t, str(txn_id) + ",", RECENT_CHARS, t))
        return cur.rowcount == 1


class AlertQueue:
    def __init__(self, c, keyed=True):
        self.c, self.keyed = c, keyed

    def enqueue(self, txn_id, score):
        self.c.execute("UPDATE alert_attempts SET n=n+1")
        if not self.keyed:                      # negative control: a queue that does not deduplicate
            self.c.execute("INSERT INTO alert_log(txn_id) VALUES(?)", (txn_id,))
            return True
        cur = self.c.execute("INSERT OR IGNORE INTO alerts VALUES(?,?)", (txn_id, score))
        return cur.rowcount == 1


class Offsets:
    def __init__(self, c, grp):
        self.c, self.grp = c, grp

    def get(self, part):
        r = self.c.execute("SELECT next_offset FROM offsets WHERE grp=? AND part=?", (self.grp, part)).fetchone()
        return 0 if r is None else r[0]

    def commit(self, part, next_offset):
        self.c.execute("INSERT INTO offsets VALUES(?,?,?) ON CONFLICT(grp,part) DO UPDATE SET next_offset=excluded.next_offset",
                       (self.grp, part, next_offset))
