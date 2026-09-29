"""Evaluation: separate RETRIEVAL quality from GENERATION quality.

If retrieval misses, no prompt can save you; if retrieval hits and the answer is wrong,
the prompt/model is at fault. Measuring both tells you where to spend effort."""
from __future__ import annotations
import json
from pathlib import Path
from . import rag


def load_golden(path="eval/golden.jsonl"):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


async def evaluate(golden, llm, retriever, s, judge: bool = False) -> dict:
    ans_q = [g for g in golden if g["answerable"]]
    hits_at_k, rr = 0, 0.0
    for g in ans_q:
        hits, _ = await retriever.search(g["question"], k=s.top_k)
        rank = next((i for i, h in enumerate(hits, 1)
                     if h.chunk.source == g["source"] and g["evidence"].lower() in h.chunk.text.lower()), None)
        hits_at_k += rank is not None
        rr += 1 / rank if rank else 0

    correct = refusal_ok = cited = faithful = 0
    rows = []
    for g in golden:
        a = await rag.answer(g["question"], llm, retriever, s)
        if g["answerable"]:
            ok = (not a.refused) and all(k.lower() in a.text.lower() for k in g["must_contain"]) if s.backend == "openai" \
                 else not a.refused
            correct += ok
            cited += bool(a.citations)
            if judge and not a.refused:
                ctx = rag.build_prompt(g["question"], a.hits)
                v = await llm.chat([{"role": "system", "content": "JUDGE. Is the ANSWER fully supported by the CONTEXT? Reply YES or NO."},
                                    {"role": "user", "content": f"{ctx}\n\nANSWER: {a.text}"}])
                faithful += (v["content"] or "").strip().upper().startswith("YES")
            rows.append((g["question"], ok))
        else:
            refusal_ok += a.refused
            rows.append((g["question"], a.refused))
    n_a, n_u = len(ans_q), len(golden) - len(ans_q)
    return {
        f"recall@{s.top_k}": hits_at_k / n_a, "mrr": rr / n_a,
        "answer_correct": correct / n_a, "citation_rate": cited / n_a,
        "refusal_accuracy": refusal_ok / n_u if n_u else None,
        "faithfulness": faithful / n_a if judge else None,
        "failures": [q for q, ok in rows if not ok],
    }
