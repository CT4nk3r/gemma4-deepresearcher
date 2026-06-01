from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentError, ResearchAgent
from .config import ConfigError, apply_overrides, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma-research",
        description="Local DeepResearch-style agent for Gemma E4B models.",
    )
    parser.add_argument("question", nargs="+", help="Research question")
    parser.add_argument("--config", help="Path to TOML config")
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "ollama", "openai", "openai-compatible", "offline"],
        help="Model provider override",
    )
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--base-url", help="Model endpoint base URL override")
    parser.add_argument(
        "--search-provider",
        choices=["duckduckgo", "none"],
        help="Search provider override",
    )
    parser.add_argument("--repo", help="Analyze a repository path in read-only mode")
    parser.add_argument("--max-iterations", type=int, help="Maximum research loop iterations")
    parser.add_argument("--trace-dir", help="Directory for JSONL traces")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace collection")
    parser.add_argument("--output", help="Write report markdown to a file")
    parser.add_argument("--json", action="store_true", help="Print structured JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()

    try:
        config = load_config(args.config)
        config = apply_overrides(
            config,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            search_provider=args.search_provider,
            max_iterations=args.max_iterations,
            trace_dir=args.trace_dir,
        )
        agent = ResearchAgent(config)
        report = agent.research(question, repo_path=args.repo, trace=not args.no_trace)
    except (AgentError, ConfigError, ValueError) as exc:
        print(f"gemma-research: error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.markdown, encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(report.markdown)
        if report.trace_path:
            print(f"\nTrace: {report.trace_path}")
    return 0
