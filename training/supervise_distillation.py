from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from training.clean_sft_dataset import main as clean_main
    from training.distill_with_lmstudio import count_jsonl, main as distill_main
    from training.validate_sft_dataset import validate_dataset
except ModuleNotFoundError:
    from clean_sft_dataset import main as clean_main
    from distill_with_lmstudio import count_jsonl, main as distill_main
    from validate_sft_dataset import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autonomously resume, clean, and validate LM Studio teacher distillation."
    )
    parser.add_argument("--input", default="data\\public_bootstrap_sft.jsonl")
    parser.add_argument("--raw-output", default="data\\teacher_distilled_starter_sft.jsonl")
    parser.add_argument("--clean-output", default="data\\teacher_distilled_clean_sft.jsonl")
    parser.add_argument("--stats-output", default="data\\teacher_distilled_clean_sft.stats.json")
    parser.add_argument("--status-output", default=".gemma-research\\distillation_status.json")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="alibaba-nlp_tongyi-deepresearch-30b-a3b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--resume-skip-count", type=int)
    parser.add_argument("--focus-file")
    parser.add_argument("--failure-sleep", type=float, default=60.0)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--max-stalled-chunks", type=int, default=5)
    parser.add_argument("--pause-file", default=".gemma-research\\distillation.pause")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.target < 1:
        raise SystemExit("--target must be at least 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")
    if args.max_stalled_chunks < 1:
        raise SystemExit("--max-stalled-chunks must be at least 1")

    failures = 0
    stalled_chunks = 0
    write_status(args, phase="starting")
    while raw_count(args.raw_output) < args.target:
        if Path(args.pause_file).exists():
            write_status(args, phase="paused", stalled_chunks=stalled_chunks)
            print(f"Pause file exists: {args.pause_file}", flush=True)
            return 0

        current = raw_count(args.raw_output)
        next_target = min(args.target, current + args.chunk_size)
        write_status(
            args,
            phase="distilling",
            next_target=next_target,
            stalled_chunks=stalled_chunks,
        )
        print(f"Distilling from {current} to {next_target} raw examples", flush=True)

        try:
            run_distillation_chunk(args, next_target)
            clean_and_validate(args)
        except Exception as exc:
            failures += 1
            write_status(
                args,
                phase="error",
                error=str(exc),
                failures=failures,
                stalled_chunks=stalled_chunks,
            )
            print(f"Supervisor failure {failures}/{args.max_failures}: {exc}", flush=True)
            if failures >= args.max_failures:
                raise
            time.sleep(args.failure_sleep)
            continue

        updated = raw_count(args.raw_output)
        if updated <= current:
            stalled_chunks += 1
            message = (
                f"No new raw examples after chunk "
                f"{stalled_chunks}/{args.max_stalled_chunks}"
            )
            write_status(
                args,
                phase="stalled",
                next_target=next_target,
                error=message,
                failures=failures,
                stalled_chunks=stalled_chunks,
            )
            print(message, flush=True)
            if stalled_chunks >= args.max_stalled_chunks:
                raise RuntimeError(message)
            time.sleep(args.failure_sleep)
            continue

        failures = 0
        stalled_chunks = 0

    clean_and_validate(args)
    write_status(args, phase="complete", stalled_chunks=stalled_chunks)
    print(f"Reached target: {raw_count(args.raw_output)} raw examples", flush=True)
    return 0


def run_distillation_chunk(args: argparse.Namespace, next_target: int) -> None:
    distill_args = [
        "--input",
        args.input,
        "--output",
        args.raw_output,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--max-examples",
        str(next_target),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--sleep",
        str(args.sleep),
        "--retries",
        str(args.retries),
        "--retry-sleep",
        str(args.retry_sleep),
        "--resume",
        "--on-error",
        "skip",
    ]
    if args.resume_skip_count is not None:
        distill_args.extend(["--resume-skip-count", str(args.resume_skip_count)])
    if args.focus_file:
        distill_args.extend(["--focus-file", args.focus_file])
    if args.dry_run:
        distill_args.append("--dry-run")
    result = distill_main(distill_args)
    if result != 0:
        raise RuntimeError(f"distill_with_lmstudio exited with code {result}")


def clean_and_validate(args: argparse.Namespace) -> None:
    clean_result = clean_main(
        [
            "--input",
            args.raw_output,
            "--output",
            args.clean_output,
        ]
    )
    if clean_result != 0:
        raise RuntimeError(f"clean_sft_dataset exited with code {clean_result}")
    stats = validate_dataset(args.clean_output)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=True), encoding="utf-8")
    if stats.issues:
        raise RuntimeError(f"clean dataset still has {len(stats.issues)} validation issue(s)")


def write_status(
    args: argparse.Namespace,
    *,
    phase: str,
    next_target: int | None = None,
    error: str | None = None,
    failures: int = 0,
    stalled_chunks: int = 0,
) -> None:
    status_path = Path(args.status_output)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    clean_stats = validate_dataset(args.clean_output) if Path(args.clean_output).exists() else None
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "target": args.target,
        "next_target": next_target,
        "raw_output": args.raw_output,
        "raw_examples": raw_count(args.raw_output),
        "clean_output": args.clean_output,
        "clean_examples": clean_stats.examples if clean_stats else 0,
        "clean_issues": len(clean_stats.issues) if clean_stats else None,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "failures": failures,
        "stalled_chunks": stalled_chunks,
        "pause_file_exists": Path(args.pause_file).exists(),
        "error": error,
    }
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def raw_count(path: str | Path) -> int:
    candidate = Path(path)
    return count_jsonl(candidate) if candidate.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
