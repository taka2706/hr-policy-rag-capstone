# Capstone — HR Policy Knowledge Assistant

Grounded Q&A over a small policy corpus (RAG), with an async batch pipeline, evaluation harness, and a bounded-autonomy agent. Built for the *Agentic AI & RAG Engineering* programme.

## Corpus
Five synthetic HR policy documents in `data/corpus/` (leave, expenses, remote work, benefits, conduct). No real PII. Drop your own `.md`/`.txt`/`.pdf` files there and re-ingest.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
make ingest                     # chunk + embed + store
make ask Q="How many sick days do I get?"
make batch                      # async, bounded, idempotent
make agent Q="If I carry over the max days, how many can I take next year?"
make eval                       # recall@k, MRR, correctness, refusal accuracy
make test                       # offline tests (BACKEND=fake), no API key needed
make runs                       # W1 evidence: 3 saved runs -> docs/runs/
```

## Layout
- `src/hello_llm.py` — W1 first API call
- `src/capstone/` — `chunking`, `store`, `retrieve` (hybrid), `rag`, `batch`, `agent`, `evalx`, `llm` (pacing/retries/cache), `cli`
- `docs/adr/` — architecture decisions · `docs/STUDY_PATH.md` — what to learn, mapped to code
- `eval/golden.jsonl` — golden questions (12 answerable, 2 unanswerable)

## Week 1 checklist
- [x] Repo + secrets discipline
- [x] `hello_llm.py`
- [ ] Run `make runs` with your real key (saves 3 outputs)
- [x] ADR-0001 (edit to match your corpus)

## Week 3: API + UI
```bash
make api      # FastAPI on :8000  (POST /ask streams NDJSON, POST /ask_batched, GET /health)
make ui       # Streamlit UI (needs the API running)
make test     # unit tests, all mocked/offline
make stress   # stress test against the running API -> docs/stress_results.md
```
Contract and decisions: `docs/adr/0003-api-contract-and-streaming.md`.

## Week 4: structured output + cost
`make compare` runs two models on the golden set (-> `docs/model_comparison.md`). Answers carry `schema_version: "v1"` and real usage/cost; see `docs/adr/0004-*`. Notebook: `week4_lab.ipynb`.
