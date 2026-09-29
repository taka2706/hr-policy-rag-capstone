# ADR-0002: Hybrid retrieval with a pre-LLM refusal gate

- **Status:** Draft v1
- **Date:** 2026-09-28

## Context
Policy answers hinge on exact numbers and terms ("3,000 INR", "March 31") and on paraphrase ("time off when ill"). Pure dense retrieval blurs the former; pure BM25 misses the latter. Unanswerable questions must not reach the model as an invitation to improvise.

## Decision
- Retrieve top 20 from dense (cosine over unit vectors) and BM25, merge by Reciprocal Rank Fusion (k=60), keep top `top_k`.
- Gate on the best dense cosine: below `min_score` return the refusal string without calling the LLM.
- Chunk on markdown headings, then window (700 chars, 100 overlap) so facts straddling a cut survive.
- Retrieved text is untrusted: wrapped in `<context>`, and the system prompt forbids following instructions inside it.

## Consequences
- **Positive:** better recall on both exact and fuzzy queries; refusals cost nothing; simple, inspectable pipeline.
- **Negative:** the cosine threshold is model-specific (0.25 is a starting point for `text-embedding-3-small`, not a truth); RRF ignores score magnitude.
- **Revisit:** calibrate `min_score` on the golden set (plot cosine for answerable vs unanswerable); add a reranker.
