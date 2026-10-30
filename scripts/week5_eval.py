"""Week 5: run prompt variants over eval/golden_v2.jsonl, score with rule checks + LLM judge.
Usage: PYTHONPATH=src python scripts/week5_eval.py [variant ...]
Writes docs/week5_results.md and eval/baseline.json (the number future RAG work must beat)."""
import asyncio, json, statistics, sys, time
from collections import defaultdict
from capstone import rag
from capstone.config import get_settings
from capstone.evalx import load_golden
from capstone.judge import DIMS, judge
from capstone.llm import LLM
from capstone.prompts import VARIANTS
from capstone.retrieve import Retriever
from capstone.store import Store


async def run_variant(name, system, golden, chunks, m, s):
    llm = LLM(s)                      # no cache: real cost/latency
    judge_llm = LLM(s)
    r = Retriever(chunks, m, llm)
    rows = []
    for g in golden:
        a = await rag.answer(g["question"], llm, r, s, system=system)
        v = await judge(g, a, rag.build_prompt(g["question"], a.hits), judge_llm)
        rule = a.refused if not g["answerable"] else (not a.refused and all(k.lower() in a.text.lower() for k in g["must_contain"]))
        rows.append(dict(q=g["question"], cat=g["category"], answer=a.text, refused=a.refused, rule_ok=bool(rule),
                         cited=bool(a.citations), cost=a.cost_usd, **v))
    return rows


def summarise(rows):
    n = len(rows)
    out = {"n": n, "rule_pass": sum(r["rule_ok"] for r in rows) / n,
           "refusal_correct": sum(r["refusal_correct"] for r in rows) / n,
           "cost_usd": sum(r["cost"] for r in rows)}
    for d in DIMS:
        out[d] = statistics.mean(r[d] for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r["rule_ok"])
    out["by_category"] = {k: sum(v) / len(v) for k, v in by.items()}
    return out


async def main():
    s = get_settings()
    chunks, m = Store(s.db_path).load()
    golden = load_golden("eval/golden_v2.jsonl")
    names = sys.argv[1:] or list(VARIANTS)
    res, detail = {}, {}
    for nm in names:
        detail[nm] = await run_variant(nm, VARIANTS[nm], golden, chunks, m, s)
        res[nm] = summarise(detail[nm])
    json.dump({"model": s.chat_model, "date": time.strftime("%Y-%m-%d"), "results": res}, open("eval/baseline.json", "w"), indent=2)
    json.dump(detail, open("eval/week5_detail.json", "w"), indent=2)
    hdr = "| Variant | Rule pass | Refusal ok | Accuracy | Grounding | Completeness | Conciseness | Typical | Challenging | Edge | Cost |\n|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [f"| {k} | {v['rule_pass']:.0%} | {v['refusal_correct']:.0%} | {v['accuracy']:.2f} | {v['grounding']:.2f} | {v['completeness']:.2f} | {v['conciseness']:.2f} | "
             f"{v['by_category'].get('typical',0):.0%} | {v['by_category'].get('challenging',0):.0%} | {v['by_category'].get('edge',0):.0%} | ${v['cost_usd']:.4f} |" for k, v in res.items()]
    fails = [f"- **{nm}** · {r['q']} → {r['answer'][:140]} _(judge: {r['reason']})_" for nm, rs in detail.items() for r in rs if not r["rule_ok"]]
    out = (f"# Week 5 results\n{time.strftime('%Y-%m-%d %H:%M')} · model `{s.chat_model}` · {len(golden)} questions · judge = same model, rubric 1-5\n\n"
           f"{hdr}\n" + "\n".join(lines) + "\n\n## Failures (rule check)\n" + ("\n".join(fails) or "none") + "\n")
    open("docs/week5_results.md", "w").write(out); print(out)

asyncio.run(main())
