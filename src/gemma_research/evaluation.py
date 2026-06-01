from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import ResearchAgent
from .config import apply_overrides, load_config


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    mode: str
    question: str
    metrics: list[str]


@dataclass(frozen=True)
class BenchmarkResult:
    id: str
    mode: str
    success: bool
    metrics: dict[str, float]
    error: str | None = None
    trace_path: str | None = None


def run_benchmark(
    tasks: list[BenchmarkTask],
    *,
    agent: ResearchAgent,
    repo_path: str | None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for task in tasks:
        try:
            report = agent.research(
                task.question,
                repo_path=repo_path if task.mode == "repo" else None,
                trace=True,
            )
            metrics = score_report(report.markdown, task.metrics)
            results.append(
                BenchmarkResult(
                    id=task.id,
                    mode=task.mode,
                    success=True,
                    metrics=metrics,
                    trace_path=report.trace_path,
                )
            )
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    id=task.id,
                    mode=task.mode,
                    success=False,
                    metrics={metric: 0.0 for metric in task.metrics},
                    error=str(exc),
                )
            )
    return results


def score_report(markdown: str, metrics: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in metrics:
        if metric == "citation_accuracy":
            scores[metric] = 1.0 if _has_citations(markdown) else 0.0
        elif metric == "hallucination_rate":
            scores[metric] = 0.0 if _has_citations(markdown) else 1.0
        elif metric == "tool_call_success_rate":
            scores[metric] = 1.0 if markdown.strip() else 0.0
        elif metric == "task_completion_rate":
            scores[metric] = 1.0 if len(markdown.strip()) > 80 else 0.0
        else:
            scores[metric] = 0.0
    return scores


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                tasks.append(
                    BenchmarkTask(
                        id=str(payload["id"]),
                        mode=str(payload["mode"]),
                        question=str(payload["question"]),
                        metrics=[str(metric) for metric in payload["metrics"]],
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid benchmark task") from exc
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gemma Research benchmark tasks.")
    parser.add_argument("--tasks", default="configs\\benchmarks.jsonl", help="Benchmark JSONL path")
    parser.add_argument("--config", help="Agent TOML config")
    parser.add_argument("--repo", default=".", help="Repository path for repo-mode tasks")
    parser.add_argument("--provider", help="Model provider override")
    parser.add_argument("--search-provider", help="Search provider override")
    parser.add_argument("--output", help="Write JSON results to a file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = apply_overrides(
        config,
        provider=args.provider,
        search_provider=args.search_provider,
    )
    tasks = load_tasks(args.tasks)
    results = run_benchmark(tasks, agent=ResearchAgent(config), repo_path=args.repo)
    payload: dict[str, Any] = {
        "results": [asdict(result) for result in results],
        "summary": summarize_results(results),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if all(result.success for result in results) else 1


def summarize_results(results: list[BenchmarkResult]) -> dict[str, float]:
    if not results:
        return {"success_rate": 0.0}
    metric_names = sorted({metric for result in results for metric in result.metrics})
    summary = {
        "success_rate": sum(1 for result in results if result.success) / len(results),
    }
    for metric in metric_names:
        values = [result.metrics.get(metric, 0.0) for result in results]
        summary[metric] = sum(values) / len(values)
    return summary


def _has_citations(markdown: str) -> bool:
    return bool(re.search(r"\[[A-Za-z0-9_.\\/\-]+\]", markdown))


if __name__ == "__main__":
    raise SystemExit(main())
