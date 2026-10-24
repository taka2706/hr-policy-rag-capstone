"""FastAPI service: /ask (streaming NDJSON), /ask_batched, /health.  Contract: docs/adr/0003.

Stream protocol (application/x-ndjson, one JSON object per line):
  {"type":"token","data":"..."}  ...repeated
  {"type":"final","data":{AskResponse}}      -- always last on success
  {"type":"error","data":"message"}          -- last, if the pipeline fails mid-stream
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import rag
from .batch import run_batch
from .config import Settings, get_settings
from .llm import LLM, Cache
from .retrieve import Retriever
from .schemas import (AskRequest, AskResponse, BatchItem, BatchRequest, BatchResponse,
                      HealthResponse, Usage)
from .store import Store

API_VERSION = "1.0"


@dataclass
class Pipeline:
    s: Settings
    llm: LLM
    retriever: Retriever


def build_pipeline() -> Pipeline:
    s = get_settings()
    chunks, m = Store(s.db_path).load()
    if not chunks:
        raise RuntimeError("store empty — run: make ingest")
    llm = LLM(s, Cache(s.db_path))
    return Pipeline(s, llm, Retriever(chunks, m, llm))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    yield


app = FastAPI(title="Capstone RAG API", version=API_VERSION, lifespan=lifespan)


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _line(type_: str, data) -> str:
    return json.dumps({"type": type_, "data": data}) + "\n"


def to_response(a: rag.Answer) -> AskResponse:
    return AskResponse(answer=a.text, citations=a.citations, refused=a.refused,
                       usage=Usage(**a.usage) if a.usage.get("model") else None)


@app.get("/health", response_model=HealthResponse)
async def health(p: Pipeline = Depends(get_pipeline)):
    return HealthResponse(api_version=API_VERSION, backend=p.s.backend, chunks=len(p.retriever.chunks))


@app.post("/ask")
async def ask(req: AskRequest, p: Pipeline = Depends(get_pipeline)):
    async def gen():
        try:
            async for kind, val in rag.answer_stream(req.question, p.llm, p.retriever, p.s):
                if kind == "token":
                    yield _line("token", val)
                else:
                    yield _line("final", to_response(val).model_dump())
        except Exception as e:                      # headers already sent: report in-band
            yield _line("error", f"{type(e).__name__}: {e}")
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/ask_batched", response_model=BatchResponse)
async def ask_batched(req: BatchRequest, p: Pipeline = Depends(get_pipeline)):
    s = Settings(**{**p.s.__dict__, "max_concurrency": req.concurrency})
    try:
        res = await run_batch(req.questions, p.llm, p.retriever, s)
    except Exception as e:
        raise HTTPException(502, f"batch failed: {type(e).__name__}")
    items = [BatchItem(question=r.question, status=r.status, answer=r.answer, citations=r.citations,
                       refused=r.refused, error=r.error, model=r.model, cost_usd=r.cost_usd) for r in res]
    ok = sum(i.status == "ok" for i in items)
    return BatchResponse(results=items, ok=ok, failed=len(items) - ok,
                         total_cost_usd=round(sum(i.cost_usd for i in items), 6))
