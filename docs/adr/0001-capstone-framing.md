# ADR-0001: Capstone Framing — HR Policy Assistant

- **Status:** Draft v1
- **Date:** 2026-09-28
- **Author:** Prakhar

## Context
Employees ask a small set of routine policy questions (leave, expenses, remote work, benefits) whose answers already sit in handbooks. A grounded Q&A assistant can answer these with citations and escalate the rest, so HR time goes to genuinely ambiguous cases.

## Decision — Solution Framing Canvas
| Box | Your answer |
|-----|-------------|
| **Inputs** | A natural-language policy question (1–2 sentences), or a file of questions for batch runs. |
| **Outputs** | A 1–4 sentence answer with bracketed citations mapping to chunk ids, or the fixed string "I don't know based on the provided documents." |
| **Tools** | OpenAI chat + embedding models; a SQLite-backed vector store; BM25 keyword index; a safe arithmetic calculator (agent mode only). |
| **Memory** | None across questions in v1 (stateless); only a prompt cache and batch-result table for idempotency. |
| **Autonomy level** | Q&A / read-only workflow. The agent mode may call `search_docs` and `calculate` in a loop capped at 6 steps; it takes no side-effecting actions. |
| **Decision boundaries** | May answer when the best cosine ≥ threshold and cite sources; must refuse below it; must never act on instructions found in documents; anything needing a judgment call goes to HR. |

## Consequences
- **Positive:** cheap deflection of routine questions; honest refusals; every claim traceable to a source; deterministic, cacheable, idempotent batch runs.
- **Negative / risks:** correctness depends on document freshness; a wrong citation erodes trust faster than good answers build it; the refusal threshold trades false refusals against hallucinations and needs tuning per embedding model.
- **Things we'll re-visit:** threshold calibration against a larger golden set; multi-turn memory; reranking with a cross-encoder; an ANN index if the corpus grows past ~100k chunks.
