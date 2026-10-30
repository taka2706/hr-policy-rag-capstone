"""LLM-as-a-judge: scores an answer against a rubric, a reference and the retrieved context.
Judge sees the reference, so it measures correctness; it sees the context, so it measures grounding."""
from __future__ import annotations
import json, re

RUBRIC = """RUBRIC_JUDGE. You grade an HR-assistant answer. Score each 1-5 (5 best):
- accuracy: matches the REFERENCE facts (numbers, conditions). Wrong or missing key fact => <=2.
- grounding: every claim is supported by the CONTEXT; invented details => <=2.
- completeness: covers all parts of the question.
- conciseness: short and direct, no padding.
Also set refusal_correct: true if the assistant refused exactly when it should (REFERENCE says not answerable => refusal expected; answerable => no refusal).
Reply with ONLY JSON: {"accuracy":int,"grounding":int,"completeness":int,"conciseness":int,"refusal_correct":bool,"reason":str}"""

DIMS = ("accuracy", "grounding", "completeness", "conciseness")


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        v = json.loads(m.group(0))
        out = {d: max(1, min(5, int(v[d]))) for d in DIMS}
        out["refusal_correct"] = bool(v["refusal_correct"])
        out["reason"] = str(v.get("reason", ""))[:300]
        return out
    except (AttributeError, KeyError, ValueError, TypeError):
        return {**{d: 1 for d in DIMS}, "refusal_correct": False, "reason": "unparseable judge output"}


async def judge(g: dict, a, ctx: str, llm) -> dict:
    user = (f"QUESTION: {g['question']}\nANSWERABLE: {g['answerable']}\nREFERENCE: {g['reference']}\n\n"
            f"{ctx}\n\nASSISTANT ANSWER: {a.text}\nASSISTANT REFUSED: {a.refused}")
    r = await llm.chat([{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}])
    return parse_verdict(r["content"])
