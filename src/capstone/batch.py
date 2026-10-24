"""Async batch pipeline: bounded concurrency + pacing + idempotent persistence + dead letters.

Little's law: in-flight = throughput x latency. With ~2s calls, 5 workers ~ 2.5 req/s.
The semaphore bounds memory/sockets; the RateLimiter (in LLM) bounds request rate;
SQLite makes reruns skip completed work (idempotency) and records failures (dead letters).
"""
from __future__ import annotations
import asyncio, hashlib, sqlite3, time
from pydantic import BaseModel
from . import rag


class Result(BaseModel):
    qid: str
    question: str
    answer: str | None = None
    citations: list[str] = []
    refused: bool = False
    status: str = "ok"              # ok | failed
    error: str | None = None
    latency_ms: int = 0
    model: str | None = None
    cost_usd: float = 0.0           # real usage-based cost (replays return the stored value)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def make_qid(q: str, s) -> str:
    return hashlib.sha256(f"{s.chat_model}|{s.top_k}|{s.min_score}|{q}".encode()).hexdigest()[:16]


async def run_batch(questions: list[str], llm, retriever, s) -> list[Result]:
    conn = sqlite3.connect(s.db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS batch_results(qid TEXT PRIMARY KEY, status TEXT, body TEXT)")
    done = {r[0]: r[1] for r in conn.execute("SELECT qid, body FROM batch_results WHERE status='ok'")}
    sem = asyncio.Semaphore(s.max_concurrency)

    async def one(q: str) -> Result:
        qid = make_qid(q, s)
        if qid in done:
            return Result.model_validate_json(done[qid])
        async with sem:
            t = time.monotonic()
            try:
                a = await rag.answer(q, llm, retriever, s)
                r = Result(qid=qid, question=q, answer=a.text, citations=a.citations, refused=a.refused,
                           model=a.model or s.chat_model, cost_usd=a.cost_usd,
                           prompt_tokens=a.usage.get("prompt_tokens", 0),
                           completion_tokens=a.usage.get("completion_tokens", 0),
                           latency_ms=int((time.monotonic() - t) * 1000))
            except Exception as e:                       # dead-letter, don't kill the batch
                r = Result(qid=qid, question=q, status="failed", error=repr(e))
        conn.execute("INSERT OR REPLACE INTO batch_results VALUES(?,?,?)", (qid, r.status, r.model_dump_json()))
        conn.commit()
        return r

    return list(await asyncio.gather(*(one(q) for q in questions)))
