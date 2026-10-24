import argparse, asyncio, json, sys
from pathlib import Path

from .agent import run_agent
from .batch import run_batch
from .chunking import chunk_document
from .config import get_settings
from .evalx import evaluate, load_golden
from .llm import LLM, Cache
from .rag import answer
from .retrieve import Retriever
from .store import Store


def read_doc(p: Path) -> str:
    if p.suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    return p.read_text()


async def ingest(s, reset):
    llm, store = LLM(s), Store(s.db_path)
    if reset:
        store.reset()
    chunks = []
    for p in sorted(Path(s.corpus_dir).glob("*")):
        if p.suffix in (".md", ".txt", ".pdf"):
            chunks += chunk_document(p.name, read_doc(p), s.chunk_size, s.chunk_overlap)
    store.add(chunks, await llm.embed([f"{c.heading}\n{c.text}" for c in chunks]))
    print(f"ingested {len(chunks)} chunks from {s.corpus_dir}")


async def boot(s):
    chunks, m = Store(s.db_path).load()
    if not chunks:
        sys.exit("store empty — run: make ingest")
    llm = LLM(s, Cache(s.db_path))
    return llm, Retriever(chunks, m, llm)


async def main_async(a):
    s = get_settings()
    if a.cmd == "ingest":
        return await ingest(s, a.reset)
    llm, r = await boot(s)
    if a.cmd == "ask":
        x = await answer(a.q, llm, r, s)
        print(x.text, "\n\ncitations:", x.citations, f"| top_cosine={x.top_cosine:.2f}",
              f"| model={x.model} cost=${x.cost_usd:.6f} tokens={x.usage.get('prompt_tokens', 0)}+{x.usage.get('completion_tokens', 0)}")
    elif a.cmd == "batch":
        qs = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]
        s.max_concurrency = a.c
        res = await run_batch(qs, llm, r, s)
        for x in res:
            print(f"[{x.status}] {x.question} -> {x.answer}")
        print(f"\n{sum(x.status=='ok' for x in res)}/{len(res)} ok | stats={llm.stats}")
    elif a.cmd == "agent":
        out = await run_agent(a.q, llm, r, s)
        print(json.dumps(out, indent=2))
    elif a.cmd == "eval":
        rep = await evaluate(load_golden(), llm, r, s, judge=a.judge)
        print(json.dumps(rep, indent=2))


def main():
    p = argparse.ArgumentParser(prog="capstone")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest"); i.add_argument("--reset", action="store_true")
    sub.add_parser("ask").add_argument("q")
    b = sub.add_parser("batch"); b.add_argument("file"); b.add_argument("-c", type=int, default=5)
    sub.add_parser("agent").add_argument("q")
    e = sub.add_parser("eval"); e.add_argument("--judge", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
