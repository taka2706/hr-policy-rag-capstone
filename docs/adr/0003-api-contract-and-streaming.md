# ADR-0003: Public API contract v1 and NDJSON streaming

- **Status:** Accepted (contract locked as v1.0)
- **Date:** 2026-10-24
- **Author:** Prakhar

## Context
The Week 2 pipeline is a library/CLI. Week 3 exposes it over HTTP with a UI. LLM calls take 1–3 s, so waiting for a full answer feels slow; and internal models (`rag.Answer` with retrieval hits, `batch.Result`) will change as RAG evolves.

## Decision
- **Endpoints:** `POST /ask` (streaming), `POST /ask_batched`, `GET /health`.
- **Public vs internal models:** `src/capstone/schemas.py` is the only contract. `rag.Answer` / `batch.Result` are mapped to `AskResponse` / `BatchItem` in `api.py`; retrieval hits and cosine scores are never exposed.
- **Streaming format:** `application/x-ndjson`, one JSON object per line: `token`* then exactly one terminal `final` (an `AskResponse`) or `error`. Chosen over SSE because it needs no special client and is trivially parsed with `httpx.iter_lines`. Mid-stream failures are reported in-band since HTTP 200 headers are already sent.
- **Validation:** question 1–1000 chars, non-blank; batch 1–50 questions, concurrency 1–10. Violations → 422 before any LLM cost.
- **Batch errors:** per-item failures are dead-lettered (`status: failed`) with HTTP 200; the batch endpoint itself fails (502) only if the runner crashes.
- **Versioning:** `api_version` is in `/health`. Additive changes (new optional fields) keep v1; renaming/removing fields or changing the stream event types requires v2 (`/v2/...`).
- **Testability:** pipeline injected via a FastAPI dependency (`get_pipeline`) so tests use stubs, no network.

## Stress findings (docs/stress_results.md, real OpenAI backend)
- All invalid inputs (empty, blank, wrong type, malformed JSON, 1001 and 100 000 chars) → 422, no LLM call.
- 10 concurrent `/ask`: 10/10 ok, p95 ≈ 4.6 s. 50 concurrent: 50/50 ok but p50 ≈ 16 s, p95 ≈ 20 s.
- The latency growth is queueing at the client-side `RateLimiter` (300 rpm; each ask makes an embed call and a chat call ⇒ ~0.2 s/call serialised), not errors. No 429s, no dropped streams.

## Consequences
- **Positive:** fast time-to-first-token on the happy path, stable client contract, cheap rejection of bad input.
- **Negative:** one process-wide rate limiter makes tail latency grow linearly with load; streamed responses are not cached; no auth or per-client rate limiting.
- **Revisit:** per-client limits + backpressure (429/`Retry-After`) rather than unbounded queueing; raise `rpm` to match the provider tier; multiple workers; auth before any public deployment.
