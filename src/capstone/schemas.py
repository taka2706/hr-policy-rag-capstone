"""PUBLIC API contract (v1). Internal models (rag.Answer, batch.Result, Hit) never cross the wire;
they are mapped to these in api.py so internals can change without breaking clients."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 1000
MAX_BATCH = 50


def _clean(q: str) -> str:
    q = q.strip()
    if not q:
        raise ValueError("question must not be blank")
    return q


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    _v = field_validator("question")(_clean)


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = []
    refused: bool = False


class BatchRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=MAX_BATCH)
    concurrency: int = Field(default=5, ge=1, le=10)

    @field_validator("questions")
    @classmethod
    def _each(cls, qs):
        out = []
        for q in qs:
            if len(q) > MAX_QUESTION_CHARS:
                raise ValueError(f"each question must be <= {MAX_QUESTION_CHARS} chars")
            out.append(_clean(q))
        return out


class BatchItem(BaseModel):
    question: str
    status: Literal["ok", "failed"]
    answer: str | None = None
    citations: list[str] = []
    refused: bool = False
    error: str | None = None


class BatchResponse(BaseModel):
    results: list[BatchItem]
    ok: int
    failed: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    api_version: str
    backend: str
    chunks: int


class ErrorResponse(BaseModel):
    detail: str
