from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator


SYSTEM_PROMPT = """You are training a smaller Gemma4 E4B model to become a careful research specialist.
Your task is to rewrite the given answer into a high-quality research-style assistant response.

Optimize for these behaviors:
1. Identify what evidence is needed before answering.
2. Compare multiple sources when available.
3. Cite every factual claim using only the provided source IDs, e.g. [S1], [S2].
4. Never invent citations, sources, numbers, dates, names, or quotes.
5. Explicitly handle uncertainty, disagreement, missing evidence, or weak evidence.
6. Avoid unsupported claims and overconfident language.
7. Prefer concise final research reports over long rambling explanations.
8. Separate evidence from interpretation.
9. If the sources are insufficient, say so clearly.
10. Do not include hidden chain-of-thought. Provide only concise reasoning summaries.

Required output style:
- Start with a direct answer.
- Then give a short evidence summary.
- Then compare or qualify the evidence if needed.
- End with a concise final conclusion.
- Use citations throughout.
- Every factual claim should have a citation.
- If no valid citation supports a claim, remove the claim or mark it as uncertain.

Do not mention that you are a teacher model.
Do not mention distillation or training.
Do not add sources that were not provided.
Do not produce uncited factual claims."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Distill researcher SFT examples with an LM Studio teacher model."
    )
    parser.add_argument("--input", required=True, help="Input SFT JSONL")
    parser.add_argument("--output", required=True, help="Output distilled SFT JSONL")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="alibaba-nlp_tongyi-deepresearch-30b-a3b")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=0, help="Retry transient LM Studio failures")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds between retries")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip already-written input examples",
    )
    parser.add_argument(
        "--resume-skip-count",
        type=int,
        help="Override how many input examples are skipped when --resume is used.",
    )
    parser.add_argument(
        "--focus-file",
        help="Optional text file with additional distillation priorities for this run.",
    )
    parser.add_argument(
        "--on-error",
        choices=["fail", "skip", "keep-original"],
        default="fail",
        help="How to handle failed or empty teacher completions",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and render prompts without calling LM Studio")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_count = count_jsonl(output_path) if args.resume and output_path.exists() else 0
    skip_count = args.resume_skip_count if args.resume_skip_count is not None else existing_count
    focus = Path(args.focus_file).read_text(encoding="utf-8").strip() if args.focus_file else ""

    examples = read_jsonl(args.input)
    if skip_count:
        examples = _skip(examples, skip_count)
    if args.max_examples is not None:
        remaining = max(args.max_examples - existing_count, 0)
        examples = _take(examples, remaining)

    count = existing_count
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for example in examples:
            try:
                distilled = distill_example_with_retries(
                    example,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    focus=focus,
                )
            except RuntimeError:
                if args.on_error == "skip":
                    print(f"Skipped example {count + 1} after distillation failure", flush=True)
                    continue
                if args.on_error == "keep-original":
                    distilled = keep_original(example, model=args.model, dry_run=args.dry_run)
                else:
                    raise
            handle.write(json.dumps(distilled, ensure_ascii=True) + "\n")
            handle.flush()
            count += 1
            print(f"Wrote example {count}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Wrote {count} distilled examples to {args.output}")
    return 0


def distill_example_with_retries(
    example: dict[str, Any],
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    dry_run: bool,
    retries: int,
    retry_sleep: float,
    focus: str = "",
) -> dict[str, Any]:
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return distill_example(
                example,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                dry_run=dry_run,
                focus=focus,
            )
        except RuntimeError as exc:
            if attempt >= attempts or not is_retryable_error(str(exc)):
                raise
            print(
                f"Transient LM Studio error on attempt {attempt}/{attempts}: {exc}",
                flush=True,
            )
            time.sleep(retry_sleep)
    raise RuntimeError("Distillation retry loop exited unexpectedly")


def distill_example(
    example: dict[str, Any],
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    dry_run: bool = False,
    focus: str = "",
) -> dict[str, Any]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Input example must contain chat messages")
    user_content = str(messages[1].get("content", ""))
    original_answer = str(messages[-1].get("content", ""))
    prompt = build_distillation_prompt(user_content, original_answer, focus=focus)
    if dry_run:
        rewritten = f"[DRY RUN] {original_answer}"
    else:
        rewritten = call_lmstudio(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    distilled = dict(example)
    distilled["messages"] = [
        dict(messages[0]),
        dict(messages[1]),
        {"role": "assistant", "content": rewritten.strip()},
    ]
    metadata = dict(distilled.get("metadata") or {})
    metadata.update(
        {
            "distilled_by": model if not dry_run else "dry-run",
            "distillation_base_url": base_url if not dry_run else None,
        }
    )
    distilled["metadata"] = metadata
    return distilled


def build_distillation_prompt(user_content: str, original_answer: str, *, focus: str = "") -> str:
    focus_block = ""
    if focus.strip():
        focus_block = (
            "\nCurrent eval feedback to optimize for:\n"
            f"{focus.strip()}\n"
        )
    return (
        "Create the assistant response for this SFT example. Return only the final "
        "assistant message, not analysis, XML, JSON, or markdown fences.\n\n"
        "Requirements:\n"
        "- Start with a direct answer, then summarize evidence, then qualify uncertainty if needed.\n"
        "- Use only the provided user payload and sources.\n"
        "- Cite every factual claim with source ids exactly as provided, such as [1] or [S2].\n"
        "- Compare sources when multiple sources are available.\n"
        "- Separate evidence from interpretation.\n"
        "- If the evidence does not support the answer, say what is missing.\n"
        "- Do not invent URLs, citations, sources, numbers, dates, names, quotes, or facts.\n"
        "- Do not mention teacher models, distillation, training, or hidden reasoning.\n"
        f"{focus_block}\n"
        f"User payload:\n{user_content}\n\n"
        f"Original answer to improve:\n{original_answer}\n"
    )


def call_lmstudio(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc
    try:
        content = str(data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LM Studio response missing assistant content") from exc
    if not content:
        raise RuntimeError("LM Studio returned an empty assistant completion")
    return content


def is_retryable_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "model unloaded" in lowered
        or "loading" in lowered
        or "timed out" in lowered
        or "connection refused" in lowered
        or "empty assistant completion" in lowered
    )


def keep_original(example: dict[str, Any], *, model: str, dry_run: bool) -> dict[str, Any]:
    kept = dict(example)
    metadata = dict(kept.get("metadata") or {})
    metadata.update(
        {
            "distilled_by": "original",
            "distillation_failed_model": model if not dry_run else "dry-run",
        }
    )
    kept["metadata"] = metadata
    return kept


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def _take(values: Iterator[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    for index, value in enumerate(values):
        if index >= limit:
            return
        yield value


def _skip(values: Iterator[dict[str, Any]], count: int) -> Iterator[dict[str, Any]]:
    for index, value in enumerate(values):
        if index < count:
            continue
        yield value


if __name__ == "__main__":
    raise SystemExit(main())
