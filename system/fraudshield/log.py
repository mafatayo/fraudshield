"""Partitioned append-only event log (the Kafka stand-in).

Events are fixed-width binary records; the offset of a record is its index in its partition
file. The topic is partitioned by a stable hash of the originating account, so all events of
one origin account are ordered inside one partition, as in Section 3.2.1.
"""
import os
import struct
import zlib

from .config import TYPES

# txn_id, step, type, amount, oldO, newO, oldD, newD, orig, dest
FMT = struct.Struct("<qiB5d12s12s")
REC = FMT.size


def partition_of(orig, n_parts):
    return zlib.crc32(orig.encode()) % n_parts


def encode(txn_id, step, typ, amount, oo, no, od, nd, orig, dest):
    return FMT.pack(txn_id, step, TYPES.index(typ), amount, oo, no, od, nd,
                    orig.encode().ljust(12, b"\0"), dest.encode().ljust(12, b"\0"))


def decode(buf):
    t, step, ty, a, oo, no, od, nd, o, d = FMT.unpack(buf)
    return {"txn_id": t, "step": step, "type": TYPES[ty], "amount": a, "oldbalanceOrg": oo,
            "newbalanceOrig": no, "oldbalanceDest": od, "newbalanceDest": nd,
            "orig": o.rstrip(b"\0").decode(), "dest": d.rstrip(b"\0").decode()}


class LogWriter:
    def __init__(self, directory, n_parts=1):
        os.makedirs(directory, exist_ok=True)
        self.n = n_parts
        self.files = [open(os.path.join(directory, f"part{p}.bin"), "wb") for p in range(n_parts)]
        self.count = [0] * n_parts

    def append(self, txn_id, step, typ, amount, oo, no, od, nd, orig, dest):
        p = partition_of(orig, self.n)
        self.files[p].write(encode(txn_id, step, typ, amount, oo, no, od, nd, orig, dest))
        self.count[p] += 1
        return p

    def close(self):
        for f in self.files:
            f.close()


class LogReader:
    def __init__(self, directory, partition):
        self.path = os.path.join(directory, f"part{partition}.bin")
        self.f = open(self.path, "rb")

    def length(self):
        return os.path.getsize(self.path) // REC

    def read_from(self, offset, max_records):
        self.f.seek(offset * REC)
        buf = self.f.read(REC * max_records)
        return [decode(buf[i:i + REC]) for i in range(0, len(buf) - REC + 1, REC)]
