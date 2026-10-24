# Model comparison
2026-10-24 17:00 · 14 golden questions · streaming, no cache

| Model | Quality | TTFT p50 | Latency p50 | Latency p95 | Prompt tok | Completion tok | Total cost | $/1k answers |
|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 100% | 1.27s | 1.54s | 7.06s | 3928 | 233 | $0.00073 | $0.052 |
| gpt-4o | 100% | 1.27s | 1.48s | 2.45s | 3928 | 233 | $0.01215 | $0.868 |

**Trade-off.** `gpt-4o` costs 16.7x `gpt-4o-mini` and is 1.0x its median latency, for a quality change of +0% on 14 golden questions. Fill in your conclusion below (is the quality delta worth the cost at your volume?).

## Conclusion
On this 14-question golden set both models scored 100% with near-identical token counts (same prompts, short grounded answers), so `gpt-4o` buys no measurable quality here while costing ~17x more. Median latency/TTFT are equal; `gpt-4o-mini`'s p95 (7s) is a single slow outlier, not a trend (n=14). **Decision: keep `gpt-4o-mini` as default**; revisit `gpt-4o` only if the golden set grows with harder multi-hop questions where mini fails. Caveats: tiny sample, keyword-based correctness, prices from `cost.PRICES` (verify against the provider's current price page).
