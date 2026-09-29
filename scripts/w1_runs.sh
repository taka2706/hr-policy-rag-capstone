#!/usr/bin/env bash
# W1 Step 2e: three saved runs as evidence. Needs a real OPENAI_API_KEY.
set -e
python src/hello_llm.py "What is our company's sick leave policy?" > docs/runs/01-no-context-hallucination.txt
python src/hello_llm.py "Why might an LLM hallucinate?"            > docs/runs/02-why-hallucinate.txt
python src/hello_llm.py "What is RAG in one sentence?"             > docs/runs/03-what-is-rag.txt
ls -la docs/runs
