"""SQLite-backed vector store. Vectors are float32 BLOBs; search is a single matmul
over the unit-normalised matrix (exact k-NN, O(N*d)) — fine to ~100k chunks.
Beyond that you'd swap in an ANN index (HNSW/IVF); the interface stays the same."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
from .chunking import Chunk


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY, source TEXT, heading TEXT, text TEXT, emb BLOB)""")

    def reset(self):
        self.conn.execute("DELETE FROM chunks"); self.conn.commit()

    def add(self, chunks: list[Chunk], vecs: np.ndarray):
        self.conn.executemany("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?)",
            [(c.id, c.source, c.heading, c.text, v.astype(np.float32).tobytes()) for c, v in zip(chunks, vecs)])
        self.conn.commit()

    def load(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self.conn.execute("SELECT id,source,heading,text,emb FROM chunks ORDER BY id").fetchall()
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        chunks = [Chunk(*r[:4]) for r in rows]
        return chunks, np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
