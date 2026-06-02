from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from training.clean_sft_dataset import main as clean_main
    from training.distill_with_lmstudio import count_jsonl
    from training.validate_sft_dataset import validate_dataset
except ModuleNotFoundError:
    from clean_sft_dataset import main as clean_main
    from distill_with_lmstudio import count_jsonl
    from validate_sft_dataset import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an unattended teach-clean-train relay without loading teacher and student together."
    )
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--input", default="data\\public_bootstrap_sft.jsonl")
    parser.add_argument("--raw-output", default="data\\teacher_distilled_starter_sft.jsonl")
    parser.add_argument("--clean-output", default="data\\teacher_distilled_clean_sft.jsonl")
    parser.add_argument("--stats-output", default="data\\teacher_distilled_clean_sft.stats.json")
    parser.add_argument("--status-output", default=".gemma-research\\autonomous_relay_status.json")
    parser.add_argument("--log-output", default=".gemma-research\\autonomous_relay.log")
    parser.add_argument("--pause-file", default=".gemma-research\\autonomous_relay.pause")
    parser.add_argument("--distillation-pause-file", default=".gemma-research\\distillation.pause")
    parser.add_argument("--teacher-target", type=int, default=1000)
    parser.add_argument("--teacher-cycle-size", type=int, default=100)
    parser.add_argument("--teacher-chunk-size", type=int, default=25)
    parser.add_argument("--teacher-model", default="alibaba-nlp_tongyi-deepresearch-30b-a3b")
    parser.add_argument("--teacher-context-length", type=int, default=4096)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--base-model", default="google/gemma-4-e4b-it")
    parser.add_argument("--venv-python", default=".venv-rocm\\Scripts\\python.exe")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--latest-adapter-dir", default="runs\\gemma4-e4b-deepresearch-lora-latest")
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--train-max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-grad-accum", type=int, default=2)
    parser.add_argument("--train-retries", type=int, default=2)
    parser.add_argument("--train-retry-sleep", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-clean-examples", type=int, default=50)
    parser.add_argument("--wait-existing-training", action="store_true")
    args = parser.parse_args(argv)

    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.teacher_cycle_size < 1:
        raise SystemExit("--teacher-cycle-size must be at least 1")
    if args.teacher_chunk_size < 1:
        raise SystemExit("--teacher-chunk-size must be at least 1")
    if args.train_steps < 1:
        raise SystemExit("--train-steps must be at least 1")
    if args.train_retries < 0:
        raise SystemExit("--train-retries must not be negative")
    if args.train_retry_sleep < 0:
        raise SystemExit("--train-retry-sleep must not be negative")

    state_paths(args)
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    log(args, f"Relay starting; deadline={deadline.isoformat()}")
    write_status(args, phase="starting", deadline=deadline)

    if args.wait_existing_training:
        wait_for_existing_training(args, deadline)

    cycle = 0
    while datetime.now(timezone.utc) < deadline and not paused(args):
        cycle += 1
        clean_and_validate(args, deadline, cycle)

        raw_examples = jsonl_count(args.raw_output)
        if raw_examples < args.teacher_target:
            next_target = min(args.teacher_target, raw_examples + args.teacher_cycle_size)
            teach(args, deadline, cycle, next_target)
            clean_and_validate(args, deadline, cycle)
        else:
            write_status(args, phase="teacher_target_reached", cycle=cycle, deadline=deadline)

        if clean_count(args.clean_output) >= args.min_clean_examples:
            train(args, deadline, cycle)
        else:
            write_status(args, phase="waiting_for_data", cycle=cycle, deadline=deadline)
            sleep_with_pause(args, 60)

    write_status(args, phase="paused" if paused(args) else "deadline_reached", cycle=cycle, deadline=deadline)
    unload_teacher(args)
    log(args, "Relay stopped")
    return 0


def state_paths(args: argparse.Namespace) -> None:
    for path in [args.status_output, args.log_output]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def wait_for_existing_training(args: argparse.Namespace, deadline: datetime) -> None:
    while datetime.now(timezone.utc) < deadline and not paused(args) and running_train_lora():
        write_status(args, phase="waiting_for_existing_training", deadline=deadline)
        log(args, "Waiting for existing train_lora.py process to finish")
        sleep_with_pause(args, 60)


def clean_and_validate(args: argparse.Namespace, deadline: datetime, cycle: int) -> None:
    write_status(args, phase="cleaning", cycle=cycle, deadline=deadline)
    result = clean_main(["--input", args.raw_output, "--output", args.clean_output])
    if result != 0:
        raise RuntimeError(f"clean_sft_dataset exited with code {result}")
    stats = validate_dataset(args.clean_output)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=True), encoding="utf-8")
    if stats.issues:
        raise RuntimeError(f"clean dataset still has {len(stats.issues)} validation issue(s)")
    write_status(args, phase="clean", cycle=cycle, deadline=deadline)


def teach(args: argparse.Namespace, deadline: datetime, cycle: int, next_target: int) -> None:
    write_status(args, phase="loading_teacher", cycle=cycle, next_target=next_target, deadline=deadline)
    Path(args.distillation_pause_file).unlink(missing_ok=True)
    run_command(args, ["lms", "server", "start"], deadline, allow_failure=True)
    run_command(
        args,
        [
            "lms",
            "load",
            args.teacher_model,
            "--identifier",
            args.teacher_model,
            "--gpu",
            "max",
            "--context-length",
            str(args.teacher_context_length),
            "-y",
        ],
        deadline,
    )
    write_status(args, phase="teaching", cycle=cycle, next_target=next_target, deadline=deadline)
    run_command(
        args,
        [
            sys.executable,
            "training\\supervise_distillation.py",
            "--input",
            args.input,
            "--raw-output",
            args.raw_output,
            "--clean-output",
            args.clean_output,
            "--stats-output",
            args.stats_output,
            "--target",
            str(next_target),
            "--chunk-size",
            str(args.teacher_chunk_size),
            "--base-url",
            args.base_url,
            "--model",
            args.teacher_model,
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--pause-file",
            args.distillation_pause_file,
        ],
        deadline,
    )
    unload_teacher(args)


def train(args: argparse.Namespace, deadline: datetime, cycle: int) -> None:
    unload_teacher(args)
    clean_examples = clean_count(args.clean_output)
    base_run_name = (
        f"gemma4-e4b-deepresearch-lora-relay-"
        f"{cycle:03d}-{clean_examples}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    attempts = args.train_retries + 1
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        output_dir = str(Path(args.runs_dir) / run_attempt_name(base_run_name, attempt, attempts))
        write_status(
            args,
            phase="training" if attempt == 1 else "training_retry",
            cycle=cycle,
            adapter_output=output_dir,
            deadline=deadline,
            error=str(last_error) if last_error else None,
        )
        try:
            train_once(args, deadline, output_dir)
        except RuntimeError as exc:
            last_error = exc
            log(args, f"Training attempt {attempt}/{attempts} failed: {exc}")
            if attempt >= attempts or paused(args) or (deadline and datetime.now(timezone.utc) >= deadline):
                raise
            unload_teacher(args)
            sleep_with_pause(args, args.train_retry_sleep)
            continue

        prune_checkpoints(output_dir)
        latest = Path(args.latest_adapter_dir)
        shutil.rmtree(latest, ignore_errors=True)
        shutil.copytree(output_dir, latest)
        prune_checkpoints(latest)
        write_status(args, phase="trained", cycle=cycle, adapter_output=output_dir, deadline=deadline)
        return


def train_once(args: argparse.Namespace, deadline: datetime, output_dir: str) -> None:
    env = dict(os.environ)
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    run_command(
        args,
        [
            args.venv_python,
            "training\\train_lora.py",
            "--model",
            args.base_model,
            "--dataset",
            args.clean_output,
            "--output-dir",
            output_dir,
            "--max-steps",
            str(args.train_steps),
            "--max-length",
            str(args.train_max_length),
            "--batch-size",
            str(args.train_batch_size),
            "--gradient-accumulation-steps",
            str(args.train_grad_accum),
            "--learning-rate",
            str(args.learning_rate),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--bf16",
            "--gradient-checkpointing",
            "--logging-steps",
            "5",
            "--save-steps",
            "0",
            "--warmup-steps",
            "0",
        ],
        deadline,
        env=env,
    )


def prune_checkpoints(path: str | Path) -> None:
    root = Path(path).resolve()
    if not root.exists():
        return
    for checkpoint in root.rglob("checkpoint-*"):
        if checkpoint.is_dir():
            resolved = checkpoint.resolve()
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to prune checkpoint outside run directory: {resolved}")
            shutil.rmtree(resolved, ignore_errors=True)


def run_attempt_name(base_run_name: str, attempt: int, attempts: int) -> str:
    if attempts <= 1:
        return base_run_name
    return f"{base_run_name}-try{attempt:02d}"


def unload_teacher(args: argparse.Namespace) -> None:
    run_command(args, ["lms", "unload", args.teacher_model], None, allow_failure=True)


def run_command(
    args: argparse.Namespace,
    command: list[str],
    deadline: datetime | None,
    *,
    env: dict[str, str] | None = None,
    allow_failure: bool = False,
) -> None:
    command_text = " ".join(command)
    log(args, f"$ {command_text}")
    with Path(args.log_output).open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat()}] $ {command_text}\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            if paused(args) or (deadline and datetime.now(timezone.utc) >= deadline):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise RuntimeError(f"Stopped command because relay paused or deadline passed: {command_text}")
            time.sleep(10)
        if process.returncode and not allow_failure:
            raise RuntimeError(f"Command failed with code {process.returncode}: {command_text}")


def running_train_lora() -> bool:
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*train_lora.py*' } | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(result.stdout.strip())


def write_status(
    args: argparse.Namespace,
    *,
    phase: str,
    cycle: int = 0,
    deadline: datetime | None = None,
    next_target: int | None = None,
    adapter_output: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cycle": cycle,
        "deadline": deadline.isoformat() if deadline else None,
        "raw_examples": jsonl_count(args.raw_output),
        "clean_examples": clean_count(args.clean_output),
        "teacher_target": args.teacher_target,
        "next_target": next_target,
        "adapter_output": adapter_output,
        "latest_adapter_dir": args.latest_adapter_dir,
        "pause_file_exists": paused(args),
        "error": error,
    }
    Path(args.status_output).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def log(args: argparse.Namespace, message: str) -> None:
    Path(args.log_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log_output).open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat()}] {message}\n")


def paused(args: argparse.Namespace) -> bool:
    return Path(args.pause_file).exists()


def sleep_with_pause(args: argparse.Namespace, seconds: int) -> None:
    for _ in range(seconds):
        if paused(args):
            return
        time.sleep(1)


def jsonl_count(path: str | Path) -> int:
    candidate = Path(path)
    return count_jsonl(candidate) if candidate.exists() else 0


def clean_count(path: str | Path) -> int:
    try:
        return jsonl_count(path)
    except OSError:
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        status = Path(".gemma-research\\autonomous_relay_status.json")
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "phase": "error",
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        raise
