# ADR-0004: Structured output, usage-based cost tracking, schema versioning

- **Status:** Accepted
- **Date:** 2026-10-24
- **Author:** Prakhar

## Context
Week 3 returned free text and had no view of what each answer cost. We need machine-checkable answers, real cost numbers, and a way to evolve internals without breaking clients.

## Decision
- **Structured output via tool-calling:** `rag.answer` forces a `submit_answer` tool call (`answer`, `cited_sources`, `refused`). The arguments are validated into `schemas.Answer` (pydantic). Malformed output is retried once, then raises (batch dead-letters it). Failed attempts are still billed and counted.
- **Streaming stays plain text** for time-to-first-token; its final event is built from the same `Answer` shape. Real usage comes from `stream_options={"include_usage": true}`.
- **Cost from real usage only:** `cost.py` converts `response.usage` tokens with a per-model price table (USD/1M tokens). Unknown models cost 0 and must be added to `PRICES`. Cache hits cost 0. Every real call is logged to SQLite `llm_usage` (model, tokens, cost, latency); batch results persist `model`, `cost_usd`, token counts.
- **Schema versioning:** every `Answer`/`AskResponse` carries `schema_version: "v1"` (a `Literal`, so other values fail validation). Additive optional fields (`usage`, `model`, `cost_usd`, `total_cost_usd`) keep v1. Renaming/removing a field, changing a type, or changing stream event types ships as v2 alongside v1 for a deprecation window. Internal models (`rag.Answer`, `batch.Result`) are still mapped in `api.py`.
- **Cost constraints:** the pre-LLM retrieval gate stays (refusals cost no completion); questions capped at 1000 chars and batches at 50 bound worst-case spend per request; default model is `gpt-4o-mini`. Model choice is justified by `docs/model_comparison.md`.

## Consequences
- **Positive:** validated outputs, auditable per-call spend, stable client contract.
- **Negative:** forced tool call adds prompt overhead and makes `/ask_batched` non-streaming-compatible with the streamed path; two code paths (tool-call vs. streamed text) to keep consistent; price table can go stale.
- **Revisit:** per-user/day budget caps with 429 on exceed; structured streaming; prompt-token reduction (fewer chunks) if cost dominates.
