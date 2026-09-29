import asyncio, os, pytest
os.environ["BACKEND"] = "fake"
from capstone.agent import run_agent, safe_eval
from capstone.batch import run_batch
from capstone.chunking import chunk_document
from capstone.config import get_settings
from capstone.evalx import evaluate, load_golden
from capstone.llm import LLM, RateLimiter, with_retries
from capstone.rag import IDK, answer
from capstone.retrieve import BM25, Retriever
from capstone.store import Store
from pathlib import Path


@pytest.fixture
def env(tmp_path):
    s = get_settings(db_path=str(tmp_path / "t.db"))
    llm = LLM(s)
    chunks = []
    for p in sorted(Path("data/corpus").glob("*.md")):
        chunks += chunk_document(p.name, p.read_text(), s.chunk_size, s.chunk_overlap)
    st = Store(s.db_path)
    st.add(chunks, asyncio.run(llm.embed([c.text for c in chunks])))
    ch, m = st.load()
    return s, llm, Retriever(ch, m, llm)


def test_chunking_keeps_headings_and_overlap():
    txt = "# A\n" + "word " * 400 + "\n# B\nshort"
    cs = chunk_document("x.md", txt, 300, 50)
    assert cs[0].heading == "A" and cs[-1].heading == "B" and len(cs) > 3
    assert all(len(c.text) <= 300 for c in cs)


def test_bm25_exact_term():
    b = BM25(["alpha beta", "gamma 42 delta", "epsilon"])
    assert b.scores("42").argmax() == 1


def test_store_roundtrip(env):
    s, llm, r = env
    assert len(r.chunks) >= 15 and r.m.shape[0] == len(r.chunks)


def test_eval_retrieval_quality(env):
    s, llm, r = env
    rep = asyncio.run(evaluate(load_golden(), llm, r, s))
    assert rep[f"recall@{s.top_k}"] >= 0.9
    assert rep["refusal_accuracy"] == 1.0


def test_refusal_gate_skips_llm(env):
    s, llm, r = env
    a = asyncio.run(answer("Does the company provide a company car?", llm, r, s))
    assert a.refused and a.text == IDK and llm.stats["calls"] == 0


def test_answer_has_citations(env):
    s, llm, r = env
    a = asyncio.run(answer("How many days of annual leave do I get per year?", llm, r, s))
    assert a.citations and not a.refused


def test_batch_idempotent_and_dead_letter(env):
    s, llm, r = env
    qs = ["How much is the home office stipend?", "What is the daily meal limit when travelling?"]
    r1 = asyncio.run(run_batch(qs, llm, r, s))
    calls = llm.stats["calls"]
    r2 = asyncio.run(run_batch(qs, llm, r, s))
    assert llm.stats["calls"] == calls and [x.answer for x in r1] == [x.answer for x in r2]

    async def boom(*a, **k): raise RuntimeError("x")
    llm.chat = boom
    bad = asyncio.run(run_batch(["a brand new annual leave question"], llm, r, s))
    assert bad[0].status == "failed"


def test_agent_loop(env):
    s, llm, r = env
    out = asyncio.run(run_agent("How many sick days do I get?", llm, r, s))
    assert out["stopped"] == "final" and out["trace"][0]["tool"] == "search_docs"


def test_calc_is_safe():
    assert safe_eval("24 - 5 * 2") == 14
    for bad in ["__import__('os')", "2**9999", "open('x')"]:
        with pytest.raises(Exception):
            safe_eval(bad)


def test_retry_gives_up_on_non_retryable():
    n = {"c": 0}
    async def f():
        n["c"] += 1; raise ValueError
    with pytest.raises(ValueError):
        asyncio.run(with_retries(f))
    assert n["c"] == 1


def test_rate_limiter_spaces_calls():
    import time
    async def go():
        rl = RateLimiter(600); t = time.monotonic()
        for _ in range(4): await rl.acquire()
        return time.monotonic() - t
    assert asyncio.run(go()) >= 0.28
