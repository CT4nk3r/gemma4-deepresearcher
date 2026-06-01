# Evaluation

Benchmark tasks live in `configs\benchmarks.jsonl`.

## Run benchmarks

```powershell
gemma-research-eval --config configs\offline.toml --provider offline --search-provider none --repo . --output results\benchmark.json
```

Use LM Studio or Ollama configs for live model evaluation.

## Metrics

| Metric | Current scoring |
|---|---|
| `citation_accuracy` | 1.0 when the report contains recognized bracket citations, otherwise 0.0. |
| `hallucination_rate` | Proxy score: 0.0 when citations exist, otherwise 1.0. |
| `tool_call_success_rate` | 1.0 when the run produces non-empty markdown. |
| `task_completion_rate` | 1.0 when the report has substantial content. |

The current metrics are lightweight automated proxies. For real model selection, pair them with manual review of citation correctness, source quality, and whether unsupported claims were made.

## Benchmark categories

- technology research;
- repository analysis;
- literature review;
- bug investigation;
- architecture review.
