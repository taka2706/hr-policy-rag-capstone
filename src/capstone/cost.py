"""Cost accounting from ACTUAL API usage (response.usage), never from guesses.

Prices are USD per 1M tokens and are assumptions to re-check against the provider's
price page; override via PRICES. Unknown models cost 0 and are flagged by `known()`.
"""
from __future__ import annotations
import sqlite3, time
from pathlib import Path

PRICES: dict[str, tuple[float, float]] = {      # model: (input $/1M, output $/1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def known(model: str) -> bool:
    return model in PRICES


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


def usage_dict(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cost_usd": cost_usd(model, prompt_tokens, completion_tokens)}


class UsageLog:
    """One row per real LLM call: model, tokens, cost, latency. Cache hits cost nothing and are not logged."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS llm_usage(
            ts REAL, model TEXT, kind TEXT, prompt_tokens INT, completion_tokens INT,
            cost_usd REAL, latency_ms INT)""")

    def record(self, u: dict, kind: str, latency_ms: int):
        self.conn.execute("INSERT INTO llm_usage VALUES(?,?,?,?,?,?,?)",
                          (time.time(), u["model"], kind, u["prompt_tokens"], u["completion_tokens"],
                           u["cost_usd"], latency_ms))
        self.conn.commit()

    def totals(self) -> list[tuple]:
        return self.conn.execute("""SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
            SUM(cost_usd) FROM llm_usage GROUP BY model""").fetchall()
