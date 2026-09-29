"""Hybrid retrieval = dense (meaning) + BM25 (exact terms), merged by Reciprocal Rank Fusion.

Why both: embeddings match paraphrases ("time off when ill" ~ "sick leave") but blur
exact tokens (IDs, numbers, names). BM25 is the opposite. RRF merges by RANK, so we
never have to compare incomparable score scales.
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .chunking import Chunk
from .llm import tokenize


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w, 0)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass
class Hit:
    chunk: Chunk
    score: float      # fused score (rank-based)
    cosine: float     # dense similarity, used for the refusal gate


class Retriever:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, llm, mode: str = "hybrid"):
        self.chunks, self.m, self.llm, self.mode = chunks, matrix, llm, mode
        self.bm25 = BM25([f"{c.heading} {c.text}" for c in chunks])

    async def search(self, query: str, k: int = 4, pool: int = 20) -> tuple[list[Hit], float]:
        qv = (await self.llm.embed([query]))[0]
        cos = self.m @ qv
        dense_order = np.argsort(-cos)[:pool]
        bm = self.bm25.scores(query)
        bm_order = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        fused: dict[int, float] = {}
        if self.mode in ("hybrid", "dense"):
            for r, i in enumerate(dense_order):
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + r)
        if self.mode in ("hybrid", "bm25"):
            for r, i in enumerate(bm_order):
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + r)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(cos[i])) for i in top], float(cos.max()) if len(cos) else 0.0
