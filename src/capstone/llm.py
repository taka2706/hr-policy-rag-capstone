"""LLM access layer: pacing, retries w/ jittered backoff, prompt cache, fake backend.

First principle: an LLM call is a slow (~1-3s), rate-limited, occasionally failing
HTTP request. Everything here exists to make that boring and safe at concurrency.
"""
from __future__ import annotations

import asyncio, hashlib, json, random, re, sqlite3, time
from pathlib import Path

import numpy as np

from .config import Settings
from .cost import UsageLog, usage_dict

# ---------- pacing ----------
class RateLimiter:
    """Evenly spaces request starts at `rpm`. Keeps us under provider limits
    instead of discovering them via 429s."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self.interval


# ---------- retries ----------
def retryable(e: Exception) -> bool:
    try:
        import openai
    except ImportError:
        return False
    return isinstance(e, (openai.RateLimitError, openai.APIConnectionError,
                          openai.APITimeoutError, openai.InternalServerError))


async def with_retries(fn, attempts=5, base=0.5, cap=20.0):
    """Exponential backoff with FULL jitter: sleep U(0, min(cap, base*2^i)).
    Jitter de-synchronises clients so retries don't stampede in lockstep."""
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if i == attempts - 1 or not retryable(e):
                raise
            await asyncio.sleep(random.uniform(0, min(cap, base * 2 ** i)))


# ---------- cache ----------
class Cache:
    """Prompt cache keyed by hash(model, messages, temperature). Deterministic
    calls (temp 0) become free and idempotent on re-run."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        r = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def set(self, k, v):
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (k, json.dumps(v)))
        self.conn.commit()


# ---------- fake backend (offline tests) ----------
_STOP = set("a an the of to in is are do does i my me we our you your what how when who which for on at be and or it".split())


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def fake_embed(texts: list[str], dim: int = 2048) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            if w not in _STOP:
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return out


def fake_chat(messages, tools) -> dict:
    if any(m["role"] == "system" and "RUBRIC_JUDGE" in (m["content"] or "") for m in messages):
        return {"role": "assistant", "content": json.dumps(
            {"accuracy": 4, "grounding": 4, "completeness": 4, "conciseness": 4, "refusal_correct": True, "reason": "fake judge"})}
    if any(m["role"] == "system" and "JUDGE" in (m["content"] or "") for m in messages):
        return {"role": "assistant", "content": "YES"}
    if tools and tools[0]["function"]["name"] == "submit_answer":
        plain = fake_chat(messages, None)["content"] or ""
        refused = plain.startswith("I don't know")
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps(
                {"answer": plain, "cited_sources": [] if refused else [1], "refused": refused})}}]}
    if tools:
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            q = [m for m in messages if m["role"] == "user"][-1]["content"]
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "search_docs", "arguments": json.dumps({"query": q})}}]}
        return {"role": "assistant", "content": "[fake agent] " + tool_msgs[-1]["content"][:200]}
    user = messages[-1]["content"] or ""
    m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n|</context>)", user, re.S)
    if not m:
        return {"role": "assistant", "content": "I don't know based on the provided documents."}
    return {"role": "assistant", "content": m.group(1).strip()[:200] + " [1]"}


# ---------- main client ----------
class LLM:
    def __init__(self, s: Settings, cache: Cache | None = None):
        self.s, self.cache = s, cache
        self.limiter = RateLimiter(s.rpm)
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        self.usage_log = UsageLog(s.db_path) if cache else None
        self.client = None
        if s.backend == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(max_retries=0, timeout=60)   # we own retries

    async def chat(self, messages: list[dict], temperature: float = 0.0, tools: list | None = None,
                   tool_choice=None) -> dict:
        """Returns the assistant message plus a `usage` entry (model, tokens, cost_usd) from the real API response."""
        key = hashlib.sha256(json.dumps([self.s.chat_model, messages, temperature, bool(tools)],
                                        sort_keys=True).encode()).hexdigest()
        if self.cache and not tools:
            hit = self.cache.get(key)
            if hit:
                self.stats["cache_hits"] += 1
                return {**hit, "usage": usage_dict(self.s.chat_model, 0, 0)}   # cached: free
        t0 = time.monotonic()
        if self.s.backend == "fake":
            out = fake_chat(messages, tools)
            out["usage"] = usage_dict(self.s.chat_model, sum(len(tokenize(m["content"] or "")) for m in messages),
                                      len(tokenize(out["content"] or json.dumps(out.get("tool_calls", "")))))
        else:
            async def call():
                await self.limiter.acquire()
                kw = dict(model=self.s.chat_model, messages=messages, temperature=temperature)
                if tools:
                    kw["tools"] = tools
                    if tool_choice:
                        kw["tool_choice"] = tool_choice
                return await self.client.chat.completions.create(**kw)
            r = await with_retries(call)
            m = r.choices[0].message
            out = {"role": "assistant", "content": m.content,
                   "usage": usage_dict(self.s.chat_model, r.usage.prompt_tokens, r.usage.completion_tokens)}
            if m.tool_calls:
                out["tool_calls"] = [{"id": t.id, "type": "function",
                                      "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                     for t in m.tool_calls]
        self._account(out["usage"], "chat", t0)
        if self.cache and not tools:
            self.cache.set(key, {k: v for k, v in out.items() if k != "usage"})
        return out

    def _account(self, u: dict, kind: str, t0: float):
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += u["prompt_tokens"]
        self.stats["completion_tokens"] += u["completion_tokens"]
        self.stats["cost_usd"] += u["cost_usd"]
        if self.usage_log:
            self.usage_log.record(u, kind, int((time.monotonic() - t0) * 1000))

    async def chat_stream(self, messages: list[dict], temperature: float = 0.0, usage_out: dict | None = None):
        """Async generator yielding text deltas. Not cached (streams are for UX, not replay).
        If `usage_out` is given it is filled with real token usage/cost once the stream ends."""
        t0 = time.monotonic()
        if self.s.backend == "fake":
            text = fake_chat(messages, None)["content"] or ""
            for w in re.findall(r"\S+\s*", text):
                await asyncio.sleep(0)
                yield w
            u = usage_dict(self.s.chat_model, sum(len(tokenize(m["content"] or "")) for m in messages),
                           len(tokenize(text)))
        else:
            await self.limiter.acquire()
            stream = await self.client.chat.completions.create(
                model=self.s.chat_model, messages=messages, temperature=temperature, stream=True,
                stream_options={"include_usage": True})      # final chunk carries real usage
            u = usage_dict(self.s.chat_model, 0, 0)
            async for ev in stream:
                if ev.usage:
                    u = usage_dict(self.s.chat_model, ev.usage.prompt_tokens, ev.usage.completion_tokens)
                if ev.choices and ev.choices[0].delta.content:
                    yield ev.choices[0].delta.content
        self._account(u, "stream", t0)
        if usage_out is not None:
            usage_out.update(u)

    async def embed(self, texts: list[str]) -> np.ndarray:
        if self.s.backend == "fake":
            v = fake_embed(texts)
        else:
            parts = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i + 100]
                async def call(b=batch):
                    await self.limiter.acquire()
                    return await self.client.embeddings.create(model=self.s.embed_model, input=b)
                r = await with_retries(call)
                parts.append(np.array([d.embedding for d in r.data], dtype=np.float32))
            v = np.vstack(parts)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)       # unit vectors: dot product == cosine
