# Study path — what to learn, mapped to the code

| Order | Read this file | Concept (first principle) |
|---|---|---|
| 1 | `src/hello_llm.py` | LLM call = stateless HTTP POST; `messages` is all the state; cost/latency scale with tokens |
| 2 | `chunking.py` | retrieval granularity = answer granularity; split on meaning, overlap to survive cuts |
| 3 | `llm.py::fake_embed`, `store.py` | embedding = text→vector; similar meaning = small angle; unit vectors make dot = cosine |
| 4 | `retrieve.py` | dense vs lexical failure modes; rank fusion avoids comparing incomparable scores |
| 5 | `rag.py` | grounding prompt, citations, refusing before generating |
| 6 | `evalx.py`, `eval/golden.jsonl` | separate retrieval from generation; recall@k, MRR; no eval = tuning by vibes |
| 7 | `llm.py` (limiter/retry/cache), `batch.py` | Little's law, pacing, jittered backoff, idempotency, dead letters |
| 8 | `agent.py` | agent = loop; bound the autonomy; tool output is untrusted; never `eval()` model strings |

Exercises that teach the most:
1. Print the raw `resp` in `hello_llm.py`; read token usage and finish_reason.
2. Set `top_k=1` then `top_k=10`; watch eval numbers move.
3. Sweep `min_score` 0.1→0.5, plot false refusals vs false answers.
4. Set `max_concurrency` to 1, 5, 20 with `rpm=60`; find where the limiter, not the semaphore, is the bottleneck.
5. Port `batch.py` to Go (errgroup + `golang.org/x/time/rate`) and compare.
