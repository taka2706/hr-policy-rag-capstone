"""Mocked tests: no network, no DB, no real LLM. Pipeline pieces are stubs."""
import json, os
os.environ["BACKEND"] = "fake"
from types import SimpleNamespace as NS

import httpx, pytest

from capstone import rag
from capstone.api import Pipeline, app
from capstone.config import get_settings
from capstone.rag import IDK


def hit(i=1):
    return NS(chunk=NS(id=f"doc.md#{i}", source="doc.md", heading="H", text="Annual leave is 24 days."),
              score=0.1, cosine=0.9)


class StubRetriever:
    def __init__(self, hits, top=0.9):
        self.hits, self.top, self.chunks = hits, top, [1, 2, 3]

    async def search(self, q, k=4):
        return self.hits, self.top


class StubLLM:
    def __init__(self, deltas=("24 days ", "[1]"), fail=False):
        self.deltas, self.fail, self.stats = deltas, fail, {"calls": 0}

    async def chat(self, messages, **k):
        if self.fail:
            raise RuntimeError("llm down")
        return {"role": "assistant", "content": "".join(self.deltas)}

    async def chat_stream(self, messages, **k):
        for d in self.deltas:
            yield d
        if self.fail:
            raise RuntimeError("llm down")


def client(llm=None, retriever=None, tmp=None):
    s = get_settings(db_path=str(tmp / "t.db")) if tmp else get_settings()
    app.state.pipeline = Pipeline(s, llm or StubLLM(), retriever or StubRetriever([hit()]))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def events(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l]


# ---- pipeline units ----
async def test_answer_stream_yields_tokens_then_final():
    out = [x async for x in rag.answer_stream("q", StubLLM(), StubRetriever([hit()]), get_settings())]
    assert [k for k, _ in out] == ["token", "token", "final"]
    assert out[-1][1].citations == ["doc.md#1"] and not out[-1][1].refused


async def test_gate_refuses_without_calling_llm():
    llm = StubLLM()
    out = [x async for x in rag.answer_stream("q", llm, StubRetriever([hit()], top=0.01), get_settings())]
    assert out[-1][1].refused and out[-1][1].text == IDK and llm.stats["calls"] == 0


# ---- endpoints ----
async def test_health():
    async with client() as c:
        r = await c.get("/health")
    assert r.status_code == 200 and r.json() == {
        "status": "ok", "api_version": "1.0", "backend": "fake", "chunks": 3}


async def test_ask_streams_and_ends_with_final():
    async with client() as c:
        r = await c.post("/ask", json={"question": "How many leave days?"})
    ev = events(r)
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert [e["type"] for e in ev] == ["token", "token", "final"]
    assert ev[-1]["data"] == {"answer": "24 days [1]", "citations": ["doc.md#1"], "refused": False}


async def test_ask_response_does_not_leak_internals():
    async with client() as c:
        ev = events(await c.post("/ask", json={"question": "x"}))
    assert set(ev[-1]["data"]) == {"answer", "citations", "refused"}


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "   "}, {"question": "a" * 1001},
                                  {"question": 5}, {"q": "x"}])
async def test_ask_rejects_invalid_input(body):
    async with client() as c:
        assert (await c.post("/ask", json=body)).status_code == 422


async def test_ask_reports_midstream_failure_in_band():
    async with client(llm=StubLLM(fail=True)) as c:
        r = await c.post("/ask", json={"question": "x"})
    assert r.status_code == 200 and events(r)[-1]["type"] == "error"


async def test_ask_batched_mixed_results(tmp_path):
    llm = StubLLM()
    async with client(llm=llm, tmp=tmp_path) as c:
        r = await c.post("/ask_batched", json={"questions": ["one?", "two?"]})
    b = r.json()
    assert r.status_code == 200 and b["ok"] == 2 and b["failed"] == 0
    assert b["results"][0]["citations"] == ["doc.md#1"]


async def test_ask_batched_failure_is_dead_lettered(tmp_path):
    async with client(llm=StubLLM(fail=True), tmp=tmp_path) as c:
        r = await c.post("/ask_batched", json={"questions": ["one?"]})
    assert r.status_code == 200 and r.json()["failed"] == 1 and r.json()["results"][0]["status"] == "failed"


@pytest.mark.parametrize("body", [{"questions": []}, {"questions": [""]}, {"questions": ["x"] * 51},
                                  {"questions": ["x" * 1001]}, {"questions": ["x"], "concurrency": 99}])
async def test_ask_batched_rejects_invalid(body):
    async with client() as c:
        assert (await c.post("/ask_batched", json=body)).status_code == 422
