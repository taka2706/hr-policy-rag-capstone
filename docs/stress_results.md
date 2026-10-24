# Stress test results
Target: `http://127.0.0.1:8000`  ·  2026-10-24 16:22

| Scenario | Outcome | Notes |
|---|---|---|
| invalid: empty body | HTTP 422 |  |
| invalid: empty question | HTTP 422 |  |
| invalid: whitespace question | HTTP 422 |  |
| invalid: wrong type | HTTP 422 |  |
| invalid: 1001 chars | HTTP 422 |  |
| invalid: 100k chars | HTTP 422 |  |
| invalid: malformed JSON | HTTP 422 |  |
| 1000 chars (limit) | HTTP 200 last=final | 1.97s |
| unicode/emoji | HTTP 200 last=final | 1.63s |
| prompt injection | HTTP 200 last=final | 0.47s |
| concurrent /ask x10 | 10/10 ok | wall 4.6s p50 4.25s p95 4.61s ttfb p50 3.75s |
| concurrent /ask x50 | 50/50 ok | wall 20.6s p50 15.86s p95 20.35s ttfb p50 15.56s |
| ask_batched x10 | HTTP 200 ok=10 failed=0 |  |
| ask_batched x51 | HTTP 422 |  |
| health | HTTP 200 |  |
