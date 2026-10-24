"""Compare two chat models on the golden set: quality vs cost vs latency (streaming TTFT).
Usage: PYTHONPATH=src python scripts/compare_models.py [modelA modelB]   -> docs/model_comparison.md
Uses real API usage for cost; no prompt cache so latency is real."""
import asyncio, statistics, sys, time
from capstone import rag
from capstone.config import get_settings
from capstone.evalx import load_golden
from capstone.llm import LLM
from capstone.retrieve import Retriever
from capstone.store import Store

MODELS = sys.argv[1:3] or ["gpt-4o-mini", "gpt-4o"]


async def run(model, golden, chunks, m):
    s = get_settings(chat_model=model)
    llm = LLM(s)
    r = Retriever(chunks, m, llm)
    ok = ttft = None
    rows = []
    for g in golden:
        t0 = time.monotonic(); first = None; fin = None
        async for kind, val in rag.answer_stream(g["question"], llm, r, s):
            if kind == "token" and first is None:
                first = time.monotonic() - t0
            elif kind == "final":
                fin = val
        total = time.monotonic() - t0
        good = fin.refused if not g["answerable"] else (not fin.refused and all(k.lower() in fin.text.lower() for k in g["must_contain"]))
        rows.append((good, first or total, total, fin.cost_usd, fin.usage.get("prompt_tokens", 0), fin.usage.get("completion_tokens", 0)))
    n = len(rows)
    p95 = sorted(x[2] for x in rows)[min(n - 1, int(.95 * n))]
    return dict(model=model, n=n, quality=sum(x[0] for x in rows) / n, ttft=statistics.median(x[1] for x in rows),
                p50=statistics.median(x[2] for x in rows), p95=p95, cost=sum(x[3] for x in rows),
                ptok=sum(x[4] for x in rows), ctok=sum(x[5] for x in rows))


async def main():
    s = get_settings()
    chunks, m = Store(s.db_path).load()
    golden = load_golden()
    res = [await run(mod, golden, chunks, m) for mod in MODELS]
    hdr = "| Model | Quality | TTFT p50 | Latency p50 | Latency p95 | Prompt tok | Completion tok | Total cost | $/1k answers |\n|---|---|---|---|---|---|---|---|---|"
    lines = [f"| {r['model']} | {r['quality']:.0%} | {r['ttft']:.2f}s | {r['p50']:.2f}s | {r['p95']:.2f}s | {r['ptok']} | {r['ctok']} | ${r['cost']:.5f} | ${r['cost']/r['n']*1000:.3f} |" for r in res]
    a, b = res
    note = (f"\n**Trade-off.** `{b['model']}` costs {b['cost']/max(a['cost'],1e-12):.1f}x `{a['model']}` "
            f"and is {b['p50']/a['p50']:.1f}x its median latency, for a quality change of {b['quality']-a['quality']:+.0%} "
            f"on {a['n']} golden questions. Fill in your conclusion below (is the quality delta worth the cost at your volume?).\n")
    out = f"# Model comparison\n{time.strftime('%Y-%m-%d %H:%M')} · {len(golden)} golden questions · streaming, no cache\n\n{hdr}\n" + "\n".join(lines) + "\n" + note
    open("docs/model_comparison.md", "w").write(out); print(out)

asyncio.run(main())
