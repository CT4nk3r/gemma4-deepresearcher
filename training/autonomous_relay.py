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
    parser.add_argument("--adaptive-raw-output", default="data\\closed_loop_teacher_distilled_sft.jsonl")
    parser.add_argument("--adaptive-seed-output", default="data\\closed_loop_seed_sft.jsonl")
    parser.add_argument("--feedback-output", default=".gemma-research\\closed_loop_feedback.json")
    parser.add_argument("--focus-output", default=".gemma-research\\closed_loop_focus.txt")
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
    parser.add_argument(
        "--no-cumulative-adapter",
        dest="cumulative_adapter",
        action="store_false",
        help="Train each relay cycle from a fresh LoRA instead of continuing the latest adapter.",
    )
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
    parser.add_argument("--closed-loop", action="store_true", help="Run teach -> train -> eval -> feedback cycles.")
    parser.add_argument("--eval-set", default="data\\eval_set.jsonl")
    parser.add_argument("--eval-output-dir", default="eval\\results")
    parser.add_argument("--eval-max-new-tokens", type=int, default=384)
    parser.add_argument("--eval-limit", type=int, help="Evaluate only the first N eval examples each cycle.")
    parser.add_argument(
        "--closed-loop-seed-count",
        type=int,
        default=8,
        help="Number of targeted feedback seed examples to prepare after each eval.",
    )
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
    if args.closed_loop_seed_count < 1:
        raise SystemExit("--closed-loop-seed-count must be at least 1")

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
        if args.closed_loop and adaptive_seed_ready(args):
            adaptive_current = jsonl_count(args.adaptive_raw_output)
            next_target = adaptive_current + args.teacher_cycle_size
            teach(args, deadline, cycle, next_target, adaptive=True)
            clean_and_validate(args, deadline, cycle)
        elif args.closed_loop and raw_examples < args.teacher_target:
            next_target = min(args.teacher_target, raw_examples + args.teacher_cycle_size)
            teach(args, deadline, cycle, next_target)
            clean_and_validate(args, deadline, cycle)
        elif args.closed_loop:
            write_status(args, phase="closed_loop_waiting_for_feedback", cycle=cycle, deadline=deadline)
        elif raw_examples < args.teacher_target:
            next_target = min(args.teacher_target, raw_examples + args.teacher_cycle_size)
            teach(args, deadline, cycle, next_target)
            clean_and_validate(args, deadline, cycle)
        else:
            write_status(args, phase="teacher_target_reached", cycle=cycle, deadline=deadline)

        if clean_count(args.clean_output) >= args.min_clean_examples:
            trained_output = train(args, deadline, cycle)
            if args.closed_loop and trained_output and not paused(args):
                evaluate_and_update_feedback(args, deadline, cycle, trained_output)
        else:
            write_status(args, phase="waiting_for_data", cycle=cycle, deadline=deadline)
            sleep_with_pause(args, 60)

    write_status(args, phase="paused" if paused(args) else "deadline_reached", cycle=cycle, deadline=deadline)
    unload_teacher(args)
    log(args, "Relay stopped")
    return 0


def state_paths(args: argparse.Namespace) -> None:
    for path in [
        args.status_output,
        args.log_output,
        args.feedback_output,
        args.focus_output,
        args.adaptive_raw_output,
        args.adaptive_seed_output,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def wait_for_existing_training(args: argparse.Namespace, deadline: datetime) -> None:
    while datetime.now(timezone.utc) < deadline and not paused(args) and running_train_lora():
        write_status(args, phase="waiting_for_existing_training", deadline=deadline)
        log(args, "Waiting for existing train_lora.py process to finish")
        sleep_with_pause(args, 60)


def clean_and_validate(args: argparse.Namespace, deadline: datetime, cycle: int) -> None:
    write_status(args, phase="cleaning", cycle=cycle, deadline=deadline)
    clean_input = combined_raw_input(args)
    result = clean_main(["--input", clean_input, "--output", args.clean_output])
    if result != 0:
        raise RuntimeError(f"clean_sft_dataset exited with code {result}")
    stats = validate_dataset(args.clean_output)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=True), encoding="utf-8")
    if stats.issues:
        raise RuntimeError(f"clean dataset still has {len(stats.issues)} validation issue(s)")
    write_status(args, phase="clean", cycle=cycle, deadline=deadline)


def combined_raw_input(args: argparse.Namespace) -> str:
    adaptive = Path(args.adaptive_raw_output)
    if not adaptive.exists() or jsonl_count(adaptive) == 0:
        return args.raw_output

    combined = Path(args.raw_output).parent / "teacher_distilled_combined_sft.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for source in [Path(args.raw_output), adaptive]:
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as source_handle:
                for line in source_handle:
                    if line.strip():
                        handle.write(line)
    return str(combined)


def teach(args: argparse.Namespace, deadline: datetime, cycle: int, next_target: int, *, adaptive: bool = False) -> None:
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
    input_path = args.adaptive_seed_output if adaptive else args.input
    raw_output = args.adaptive_raw_output if adaptive else args.raw_output
    resume_skip_count = jsonl_count(raw_output) if adaptive else None
    write_status(args, phase="teaching", cycle=cycle, next_target=next_target, deadline=deadline)
    command = [
            sys.executable,
            "training\\supervise_distillation.py",
            "--input",
            input_path,
            "--raw-output",
            raw_output,
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
    ]
    if adaptive:
        command.extend(["--resume-skip-count", str(resume_skip_count)])
    if Path(args.focus_output).exists():
        command.extend(["--focus-file", args.focus_output])
    run_command(args, command, deadline)
    unload_teacher(args)


def train(args: argparse.Namespace, deadline: datetime, cycle: int) -> str | None:
    unload_teacher(args)
    clean_examples = clean_count(args.clean_output)
    resume_adapter = latest_adapter_for_resume(args)
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
            resume_adapter=resume_adapter,
            deadline=deadline,
            error=str(last_error) if last_error else None,
        )
        try:
            train_once(args, deadline, output_dir, resume_adapter=resume_adapter)
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
        write_status(
            args,
            phase="trained",
            cycle=cycle,
            adapter_output=output_dir,
            resume_adapter=resume_adapter,
            deadline=deadline,
        )
        return output_dir
    return None


def train_once(
    args: argparse.Namespace,
    deadline: datetime,
    output_dir: str,
    *,
    resume_adapter: str | None = None,
) -> None:
    env = dict(os.environ)
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    command = [
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
    ]
    if resume_adapter:
        command.extend(["--resume-adapter", resume_adapter])
    run_command(
        args,
        command,
        deadline,
        env=env,
    )


def evaluate_and_update_feedback(
    args: argparse.Namespace,
    deadline: datetime,
    cycle: int,
    adapter_output: str,
) -> None:
    write_status(
        args,
        phase="evaluating",
        cycle=cycle,
        adapter_output=adapter_output,
        deadline=deadline,
    )
    before = latest_eval_json(args.eval_output_dir)
    command = [
        args.venv_python,
        "training\\eval_adapter.py",
        "--adapter",
        args.latest_adapter_dir,
        "--base-model",
        args.base_model,
        "--eval-set",
        args.eval_set,
        "--output-dir",
        args.eval_output_dir,
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
    ]
    if args.eval_limit is not None:
        command.extend(["--limit", str(args.eval_limit)])
    run_command(args, command, deadline, env=rocm_env())

    result_path = latest_eval_json(args.eval_output_dir)
    if result_path is None or result_path == before:
        raise RuntimeError("Evaluation did not write a new result file")
    feedback = build_feedback(result_path, seed_count=args.closed_loop_seed_count)
    Path(args.feedback_output).write_text(
        json.dumps(feedback, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    Path(args.focus_output).write_text(feedback["focus_text"], encoding="utf-8")
    Path(args.adaptive_raw_output).unlink(missing_ok=True)
    write_adaptive_seed_examples(feedback, args.adaptive_seed_output)
    write_status(
        args,
        phase="feedback_ready",
        cycle=cycle,
        adapter_output=adapter_output,
        deadline=deadline,
    )


def latest_eval_json(output_dir: str | Path) -> Path | None:
    candidates = sorted(Path(output_dir).glob("eval-*.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def build_feedback(eval_json: str | Path, *, seed_count: int) -> dict[str, Any]:
    payload = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    delta = aggregate.get("delta", {})
    items = payload.get("items", [])
    worst = sorted(
        (item for item in items if isinstance(item, dict)),
        key=feedback_sort_key,
        reverse=True,
    )[:seed_count]

    priorities = []
    if float(delta.get("hallucination_proxy", 0.0)) > 0:
        priorities.append("Reduce hallucination_proxy: remove or qualify uncited factual claims.")
    if float(delta.get("citation_rate", 0.0)) < 0:
        priorities.append("Improve citation_rate: every factual claim needs a provided citation.")
    if float(delta.get("format_score", 0.0)) < 0:
        priorities.append("Restore format compliance: direct answer, evidence summary, conclusion.")
    if float(delta.get("uncertainty_score", 0.0)) < 0:
        priorities.append("Restore uncertainty handling when evidence is missing or weak.")
    if not priorities:
        priorities.append("Maintain gains while reducing unsupported factual additions.")

    focus_text = "\n".join(
        [
            "Closed-loop eval feedback:",
            f"- overall_score delta: {delta.get('overall_score', 0.0):+}",
            f"- citation_rate delta: {delta.get('citation_rate', 0.0):+}",
            f"- hallucination_proxy delta: {delta.get('hallucination_proxy', 0.0):+} (lower is better)",
            f"- uncertainty_score delta: {delta.get('uncertainty_score', 0.0):+}",
            f"- format_score delta: {delta.get('format_score', 0.0):+}",
            "Priorities:",
            *[f"- {priority}" for priority in priorities],
        ]
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_result": str(eval_json),
        "aggregate": aggregate,
        "priorities": priorities,
        "focus_text": focus_text,
        "worst_items": worst,
    }


def feedback_sort_key(item: dict[str, Any]) -> tuple[float, float]:
    delta = item.get("delta", {})
    hallucination_regression = float(delta.get("hallucination_proxy", 0.0))
    overall_drop = -float(delta.get("overall_score", 0.0))
    return hallucination_regression, overall_drop


def write_adaptive_seed_examples(feedback: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in feedback.get("worst_items", []):
            example = adaptive_seed_example(item, feedback.get("priorities", []))
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")


def adaptive_seed_example(item: dict[str, Any], priorities: list[str]) -> dict[str, Any]:
    question = str(item.get("question", "")).strip()
    adapter_response = str(((item.get("adapter") or {}).get("response") or "")).strip()
    user_payload = (
        f"Question:\n{question}\n\n"
        "Available source notes:\n"
        "[S1] The prompt does not provide primary source documents, measurements, or named studies.\n"
        "[S2] A careful research answer should describe what evidence would be needed and avoid unsupported factual additions.\n"
        "[S3] Any claim not supported by the provided notes should be removed, qualified, or marked as insufficiently evidenced.\n\n"
        "Current repair priorities:\n"
        + "\n".join(f"- {priority}" for priority in priorities)
    )
    return {
        "messages": [
            {"role": "system", "content": "You are a careful research assistant."},
            {"role": "user", "content": user_payload},
            {"role": "assistant", "content": adapter_response},
        ],
        "metadata": {
            "dataset_id": "closed-loop-eval-repair",
            "source_eval_index": item.get("index"),
            "repair_target": "hallucination_proxy",
        },
    }


def adaptive_seed_ready(args: argparse.Namespace) -> bool:
    seed = Path(args.adaptive_seed_output)
    return seed.exists() and jsonl_count(seed) > jsonl_count(args.adaptive_raw_output)


def latest_adapter_for_resume(args: argparse.Namespace) -> str | None:
    if not getattr(args, "cumulative_adapter", True):
        return None
    latest = Path(args.latest_adapter_dir)
    if adapter_ready(latest):
        return str(latest)
    return None


def adapter_ready(path: str | Path) -> bool:
    adapter = Path(path)
    return (
        adapter.is_dir()
        and (adapter / "adapter_config.json").is_file()
        and (adapter / "adapter_model.safetensors").is_file()
    )


def rocm_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    return env


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
    resume_adapter: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cycle": cycle,
        "deadline": deadline.isoformat() if deadline else None,
        "raw_examples": jsonl_count(args.raw_output),
        "adaptive_raw_examples": jsonl_count(getattr(args, "adaptive_raw_output", "")),
        "adaptive_seed_examples": jsonl_count(getattr(args, "adaptive_seed_output", "")),
        "clean_examples": clean_count(args.clean_output),
        "teacher_target": args.teacher_target,
        "next_target": next_target,
        "adapter_output": adapter_output,
        "resume_adapter": resume_adapter,
        "latest_adapter_dir": args.latest_adapter_dir,
        "feedback_output": getattr(args, "feedback_output", None),
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
    if not path:
        return 0
    candidate = Path(path)
    return count_jsonl(candidate) if candidate.exists() and candidate.is_file() else 0


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
