from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from training.train_lora import (
        _install_transformers_continuous_batching_shim,
        _model_loader_for_config,
        _reject_gguf_model,
    )
except ModuleNotFoundError:
    from train_lora import (  # type: ignore[no-redef]
        _install_transformers_continuous_batching_shim,
        _model_loader_for_config,
        _reject_gguf_model,
    )


DEFAULT_BASE_MODEL = "google/gemma-4-e4b-it"
DEFAULT_ADAPTER = "runs\\gemma4-e4b-deepresearch-lora-latest"
DEFAULT_EVAL_SET = "data\\eval_set.jsonl"
DEFAULT_OUTPUT_DIR = "eval\\results"

EVAL_SYSTEM_PROMPT = """You are a careful research-specialist assistant.

Required response structure:
Direct answer: give the shortest supported answer first.
Evidence summary: summarize what evidence supports the answer.
Conclusion: finish with a concise bottom line.

Rules:
- Cite factual claims with bracket citations such as [S1] or [1] when sources are available.
- Do not invent citations or source details.
- If the prompt does not provide enough evidence, say that clearly.
- Separate evidence from interpretation or opinion.
- Use cautious language for uncertain, contested, or missing evidence.
- Do not reveal hidden chain-of-thought."""

_CITATION_RE = re.compile(r"\[(?:S\d+|\d+)\]", re.IGNORECASE)
_UNCERTAINTY_RE = re.compile(
    r"\b("
    r"unclear|uncertain|insufficient|not enough evidence|limited evidence|"
    r"cannot determine|can't determine|cannot verify|can't verify|unknown|"
    r"not established|not proven|appears|seems|suggests|may|might|could|"
    r"confidence|tentative|more evidence|additional evidence|source documents"
    r")\b",
    re.IGNORECASE,
)
_CLAIM_VERB_RE = re.compile(
    r"\b("
    r"is|are|was|were|has|have|had|can|caused|causes|causing|shows|showed|"
    r"indicates|indicated|suggests|suggested|increases|decreases|reduces|"
    r"supports|depends|requires|means|leads|led|reported|found"
    r")\b",
    re.IGNORECASE,
)
_KNOWN_ENTITY_RE = re.compile(
    r"\b(?:AI|API|CPU|EU|FDA|GPU|HF|LLM|NASA|ROCm|SFT|UK|US|WHO|COVID|mRNA|LoRA)\b"
)
_CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


@dataclass(frozen=True)
class EvalExample:
    question: str
    expected_behaviors: list[str]


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    torch: Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Gemma4 DeepResearch LoRA adapter against the base model."
    )
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="PEFT adapter directory or Hub id")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Trainable HF safetensors model id")
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET, help="JSONL eval questions")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON and markdown results")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", help="Device override such as cuda or cpu; ignored when --device-map is set")
    parser.add_argument("--device-map", help="Optional Transformers device_map, for example auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples")
    parser.add_argument("--dry-run", action="store_true", help="Exercise scoring and output without loading models")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Load base and adapter in one Python process instead of isolated subprocesses.",
    )
    parser.add_argument("--generate-only", choices=["base", "adapter"], help=argparse.SUPPRESS)
    parser.add_argument("--responses-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    examples = read_eval_set(args.eval_set)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No eval examples found in {args.eval_set}")

    if args.generate_only:
        if not args.responses_output:
            raise SystemExit("--responses-output is required with --generate-only")
        responses = generate_variant_responses(args.generate_only, args, examples)
        Path(args.responses_output).write_text(
            json.dumps({"variant": args.generate_only, "responses": responses}, ensure_ascii=True),
            encoding="utf-8",
        )
        return 0

    payload = run_evaluation(args, examples)
    json_path, markdown_path = write_results(payload, args.output_dir)
    print(render_stdout_table(payload))
    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote markdown: {markdown_path}")
    return 0


def read_eval_set(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, value in read_jsonl(path):
        question = value.get("question")
        expected = value.get("expected_behaviors")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{line_number}: question must be a non-empty string")
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise ValueError(f"{path}:{line_number}: expected_behaviors must be a list of strings")
        examples.append(
            EvalExample(
                question=question.strip(),
                expected_behaviors=[item.strip() for item in expected if item.strip()],
            )
        )
    return examples


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
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
            yield line_number, value


def run_evaluation(args: argparse.Namespace, examples: list[EvalExample]) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    validate_model_inputs(args.base_model, args.adapter, skip_adapter_path=args.dry_run)

    if args.dry_run or args.in_process:
        base_responses = generate_variant_responses("base", args, examples)
        adapter_responses = generate_variant_responses("adapter", args, examples)
    else:
        base_responses = generate_variant_responses_in_subprocess("base", args)
        adapter_responses = generate_variant_responses_in_subprocess("adapter", args)

    items: list[dict[str, Any]] = []
    for index, (example, base_response, adapter_response) in enumerate(
        zip(examples, base_responses, adapter_responses), start=1
    ):
        base_metrics = score_response(base_response, example.expected_behaviors)
        adapter_metrics = score_response(adapter_response, example.expected_behaviors)
        items.append(
            {
                "index": index,
                "question": example.question,
                "expected_behaviors": example.expected_behaviors,
                "base": {"response": base_response, "metrics": base_metrics},
                "adapter": {"response": adapter_response, "metrics": adapter_metrics},
                "delta": metric_delta(base_metrics, adapter_metrics),
            }
        )

    aggregate = {
        "base": aggregate_metrics(items, "base"),
        "adapter": aggregate_metrics(items, "adapter"),
    }
    aggregate["delta"] = metric_delta(aggregate["base"], aggregate["adapter"])
    return {
        "timestamp": timestamp,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "eval_set": args.eval_set,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "dry_run": args.dry_run,
        },
        "aggregate": aggregate,
        "items": items,
    }


def validate_model_inputs(base_model: str, adapter: str, *, skip_adapter_path: bool = False) -> None:
    _reject_gguf_model(base_model)
    adapter_path = Path(adapter)
    if adapter_path.suffix.lower() == ".gguf":
        raise SystemExit("Adapters must be PEFT LoRA directories or Hub ids, not GGUF files.")
    if skip_adapter_path:
        return
    if adapter_path.exists():
        validate_adapter_dir(adapter_path)
    elif "\\" in adapter or "/" in adapter:
        raise SystemExit(f"Adapter directory not found: {adapter}")


def validate_adapter_dir(path: str | Path) -> None:
    adapter_path = Path(path)
    missing = [
        name
        for name in ["adapter_config.json", "adapter_model.safetensors"]
        if not (adapter_path / name).exists()
    ]
    if missing:
        raise SystemExit(f"Adapter directory is missing required file(s): {', '.join(missing)}")


def generate_variant_responses(
    variant: str, args: argparse.Namespace, examples: list[EvalExample]
) -> list[str]:
    if args.dry_run:
        return [dry_run_response(example.question, variant) for example in examples]

    adapter = args.adapter if variant == "adapter" else None
    bundle = load_model_bundle(
        base_model=args.base_model,
        adapter=adapter,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        responses = []
        for index, example in enumerate(examples, start=1):
            print(f"{variant}: generating {index}/{len(examples)}", flush=True)
            responses.append(
                generate_response(
                    bundle,
                    example.question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            )
        return responses
    finally:
        release_model_bundle(bundle)


def generate_variant_responses_in_subprocess(variant: str, args: argparse.Namespace) -> list[str]:
    with tempfile.TemporaryDirectory(prefix=f"gemma-eval-{variant}-") as tmp:
        output_path = Path(tmp) / f"{variant}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--adapter",
            args.adapter,
            "--base-model",
            args.base_model,
            "--eval-set",
            args.eval_set,
            "--output-dir",
            args.output_dir,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--dtype",
            args.dtype,
            "--generate-only",
            variant,
            "--responses-output",
            str(output_path),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.device_map:
            command.extend(["--device-map", args.device_map])
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        env = dict(os.environ)
        env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        result = subprocess.run(command, env=env, check=False)
        if result.returncode:
            raise SystemExit(f"{variant} generation failed with exit code {result.returncode}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        responses = payload.get("responses")
        if not isinstance(responses, list) or not all(isinstance(item, str) for item in responses):
            raise SystemExit(f"{variant} generation did not write a valid response payload")
        return responses


def load_model_bundle(
    *,
    base_model: str,
    adapter: str | None,
    dtype: str,
    device: str | None,
    device_map: str | None,
    trust_remote_code: bool,
) -> ModelBundle:
    _install_transformers_continuous_batching_shim()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Adapter eval requires optional packages: python -m pip install -e ".[training]"'
        ) from exc

    tokenizer_source = tokenizer_source_for(base_model, adapter)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model_loader = _model_loader_for_config(config, AutoModelForCausalLM, AutoModelForImageTextToText)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype_for(torch, dtype),
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    model = model_loader.from_pretrained(base_model, **model_kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    if not device_map:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return ModelBundle(tokenizer=tokenizer, model=model, torch=torch)


def tokenizer_source_for(base_model: str, adapter: str | None) -> str:
    if adapter is None:
        return base_model
    adapter_path = Path(adapter)
    if adapter_path.exists() and (
        (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    ):
        return str(adapter_path)
    return base_model


def torch_dtype_for(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def generate_response(
    bundle: ModelBundle,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    tokenizer = bundle.tokenizer
    model = bundle.model
    torch = bundle.torch
    prompt = render_chat_prompt(tokenizer, build_messages(question))
    encoded = tokenizer(prompt, return_tensors="pt")
    target_device = first_model_device(model)
    encoded = {key: value.to(target_device) for key, value in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)
    prompt_tokens = encoded["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        return "\n".join(f"{message['role'].title()}: {message['content']}" for message in messages) + "\nAssistant:"


def first_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def release_model_bundle(bundle: ModelBundle) -> None:
    torch = bundle.torch
    del bundle.model
    del bundle.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dry_run_response(question: str, variant: str) -> str:
    if variant == "base":
        return (
            "Direct answer: This dry-run base response gives a generic answer to the "
            f"question without enough support.\n\n"
            "Evidence summary: It includes a factual-sounding claim but does not separate "
            "evidence from interpretation.\n\n"
            "Conclusion: More checking is needed."
        )
    return (
        "Direct answer: The answer is unclear without source documents.\n\n"
        "Evidence summary: The prompt does not provide source texts, so factual claims "
        "should be treated as insufficiently supported.\n\n"
        "Conclusion: Use additional evidence before making a firm research conclusion."
    )


def score_response(response: str, expected_behaviors: list[str]) -> dict[str, Any]:
    sentences = split_sentences(response)
    claims = [sentence for sentence in sentences if looks_like_factual_claim(sentence)]
    citation_count = len(_CITATION_RE.findall(response))
    cited_claims = [sentence for sentence in claims if _CITATION_RE.search(sentence)]
    claim_count = len(claims)
    cited_claim_count = len(cited_claims)
    uncited_claim_count = max(claim_count - cited_claim_count, 0)
    citation_rate = cited_claim_count / claim_count if claim_count else 1.0
    hallucination_proxy = (
        uncited_claim_count / claim_count if citation_count and claim_count else 0.0
    )
    uncertainty_required = requires_uncertainty(expected_behaviors)
    uncertainty_present = bool(_UNCERTAINTY_RE.search(response))
    uncertainty_score = 1.0 if not uncertainty_required or uncertainty_present else 0.0
    format_metrics = score_format(response)
    overall_score = (
        0.40 * citation_rate
        + 0.25 * (1.0 - hallucination_proxy)
        + 0.20 * uncertainty_score
        + 0.15 * format_metrics["format_score"]
    )
    return {
        "overall_score": round(overall_score, 4),
        "citation_rate": round(citation_rate, 4),
        "citation_count": citation_count,
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "uncited_claim_count": uncited_claim_count,
        "hallucination_proxy": round(hallucination_proxy, 4),
        "uncertainty_required": uncertainty_required,
        "uncertainty_present": uncertainty_present,
        "uncertainty_score": uncertainty_score,
        "format_score": format_metrics["format_score"],
        "format_parts": format_metrics["parts"],
        "claim_samples": claims[:5],
    }


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [normalize_sentence(piece) for piece in pieces if normalize_sentence(piece)]


def normalize_sentence(sentence: str) -> str:
    stripped = sentence.strip()
    stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", stripped)
    stripped = re.sub(r"^\s*#{1,6}\s*", "", stripped)
    for label in ["Direct answer:", "Evidence summary:", "Conclusion:", "Final conclusion:", "Answer:"]:
        if stripped.lower().startswith(label.lower()):
            stripped = stripped[len(label) :].strip()
    return stripped


def looks_like_factual_claim(sentence: str) -> bool:
    text = normalize_sentence(sentence)
    if not text:
        return False
    lower = text.lower().strip(":")
    if lower in {"direct answer", "evidence summary", "evidence", "conclusion", "final conclusion"}:
        return False
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 4:
        return False

    has_citation = bool(_CITATION_RE.search(text))
    has_number = bool(_NUMBER_RE.search(text))
    has_entity = bool(_KNOWN_ENTITY_RE.search(text) or _CAPITALIZED_PHRASE_RE.search(text))
    has_claim_verb = bool(_CLAIM_VERB_RE.search(text))
    uncertainty_only = bool(_UNCERTAINTY_RE.search(text)) and not (has_citation or has_number or has_entity)
    if uncertainty_only:
        return False
    return has_citation or has_number or has_entity or has_claim_verb


def requires_uncertainty(expected_behaviors: list[str]) -> bool:
    joined = " ".join(expected_behaviors).lower()
    return any(
        phrase in joined
        for phrase in [
            "uncertain",
            "uncertainty",
            "unclear",
            "insufficient",
            "limited evidence",
            "missing evidence",
            "caveat",
            "confidence",
        ]
    )


def score_format(response: str) -> dict[str, Any]:
    lower = response.lower()
    positions = {
        "direct_answer": first_index(lower, ["direct answer:", "answer:"]),
        "evidence_summary": first_index(lower, ["evidence summary:", "evidence:", "sources:"]),
        "conclusion": first_index(lower, ["conclusion:", "final conclusion:", "bottom line:"]),
    }
    has_direct = positions["direct_answer"] != -1 and positions["direct_answer"] < 500
    has_evidence = positions["evidence_summary"] != -1
    has_conclusion = positions["conclusion"] != -1
    ordered = (
        has_direct
        and has_evidence
        and has_conclusion
        and positions["direct_answer"] < positions["evidence_summary"] < positions["conclusion"]
    )
    score = sum([has_direct, has_evidence, has_conclusion, ordered]) / 4
    return {
        "format_score": round(score, 4),
        "parts": {
            "direct_answer": has_direct,
            "evidence_summary": has_evidence,
            "conclusion": has_conclusion,
            "ordered": ordered,
        },
    }


def first_index(text: str, needles: list[str]) -> int:
    positions = [text.find(needle) for needle in needles if text.find(needle) != -1]
    return min(positions) if positions else -1


def metric_delta(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, float]:
    keys = ["overall_score", "citation_rate", "hallucination_proxy", "uncertainty_score", "format_score"]
    return {key: round(float(adapter.get(key, 0.0)) - float(base.get(key, 0.0)), 4) for key in keys}


def aggregate_metrics(items: list[dict[str, Any]], side: str) -> dict[str, Any]:
    keys = ["overall_score", "citation_rate", "hallucination_proxy", "uncertainty_score", "format_score"]
    aggregate = {
        key: round(sum(float(item[side]["metrics"][key]) for item in items) / len(items), 4)
        for key in keys
    }
    aggregate["claim_count"] = sum(int(item[side]["metrics"]["claim_count"]) for item in items)
    aggregate["citation_count"] = sum(int(item[side]["metrics"]["citation_count"]) for item in items)
    aggregate["uncited_claim_count"] = sum(
        int(item[side]["metrics"]["uncited_claim_count"]) for item in items
    )
    return aggregate


def write_results(payload: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_name = f"eval-{payload['timestamp']}"
    json_path = output_path / f"{base_name}.json"
    markdown_path = output_path / f"{base_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def render_stdout_table(payload: dict[str, Any], *, include_title: bool = True) -> str:
    lines = []
    if include_title:
        lines.extend(["# Gemma4 adapter eval", ""])
    lines.extend(
        [
            "| Metric | Base | Adapter | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ["overall_score", "citation_rate", "hallucination_proxy", "uncertainty_score", "format_score"]:
        lines.append(
            f"| {key} | {fmt(payload['aggregate']['base'][key])} | "
            f"{fmt(payload['aggregate']['adapter'][key])} | {fmt_delta(payload['aggregate']['delta'][key])} |"
        )
    lines.extend(["", "| # | Question | Base | Adapter | Delta | Cite B->A | Uncited B->A | Fmt B->A |"])
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for item in payload["items"]:
        base = item["base"]["metrics"]
        adapter = item["adapter"]["metrics"]
        lines.append(
            f"| {item['index']} | {escape_table(shorten(item['question'], 52))} | "
            f"{fmt(base['overall_score'])} | {fmt(adapter['overall_score'])} | "
            f"{fmt_delta(item['delta']['overall_score'])} | "
            f"{fmt(base['citation_rate'])}->{fmt(adapter['citation_rate'])} | "
            f"{base['uncited_claim_count']}->{adapter['uncited_claim_count']} | "
            f"{fmt(base['format_score'])}->{fmt(adapter['format_score'])} |"
        )
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Gemma4 DeepResearch Adapter Evaluation",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Base model: `{payload['base_model']}`",
        f"- Adapter: `{payload['adapter']}`",
        f"- Eval set: `{payload['eval_set']}`",
        f"- Dry run: `{payload['generation']['dry_run']}`",
        "",
        "## Summary",
        "",
        render_stdout_table(payload, include_title=False),
        "",
        "## Per-question Details",
    ]
    for item in payload["items"]:
        lines.extend(
            [
                "",
                f"### {item['index']}. {item['question']}",
                "",
                f"Expected behaviors: {', '.join(item['expected_behaviors'])}",
                "",
                "| Side | Overall | Citation rate | Hallucination proxy | Uncertainty | Format |",
                "|---|---:|---:|---:|---:|---:|",
                metrics_row("Base", item["base"]["metrics"]),
                metrics_row("Adapter", item["adapter"]["metrics"]),
                "",
                "Base response:",
                "",
                fenced(item["base"]["response"]),
                "",
                "Adapter response:",
                "",
                fenced(item["adapter"]["response"]),
            ]
        )
    return "\n".join(lines) + "\n"


def metrics_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {fmt(metrics['overall_score'])} | {fmt(metrics['citation_rate'])} | "
        f"{fmt(metrics['hallucination_proxy'])} | {fmt(metrics['uncertainty_score'])} | "
        f"{fmt(metrics['format_score'])} |"
    )


def fenced(text: str) -> str:
    return "```text\n" + text.strip().replace("```", "` ` `") + "\n```"


def fmt(value: Any) -> str:
    return f"{float(value):.2f}"


def fmt_delta(value: Any) -> str:
    numeric = float(value)
    return f"{numeric:+.2f}"


def shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(width - 3, 0)].rstrip() + "..."


def escape_table(text: str) -> str:
    return text.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
