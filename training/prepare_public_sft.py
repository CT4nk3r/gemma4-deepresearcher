from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SYSTEM_PROMPT = (
    "You are a local DeepResearch agent. Plan searches, use evidence before answering, "
    "cite source ids in square brackets, and say when evidence is insufficient."
)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    converter: str
    hf_id: str | None = None
    url: str | None = None
    format: str = "hf"
    config: str | None = None
    splits: tuple[str, ...] = ("train",)
    default_max_examples: int | None = None
    purpose: str = ""
    license_note: str = ""
    trust_remote_code: bool = False


def load_manifest(path: str | Path) -> list[DatasetSpec]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    specs: list[DatasetSpec] = []
    for item in payload.get("datasets", []):
        specs.append(
            DatasetSpec(
                id=str(item["id"]),
                hf_id=item.get("hf_id"),
                url=item.get("url"),
                format=str(item.get("format", "hf")),
                config=item.get("config"),
                splits=tuple(item.get("splits", ["train"])),
                converter=str(item["converter"]),
                default_max_examples=item.get("default_max_examples"),
                purpose=str(item.get("purpose", "")),
                license_note=str(item.get("license_note", "")),
                trust_remote_code=bool(item.get("trust_remote_code", False)),
            )
        )
    return specs


def convert_row(
    row: dict[str, Any],
    *,
    converter: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metadata = dict(metadata or {})
    if converter == "webgpt_comparisons":
        return _convert_webgpt_comparison(row, metadata)
    if converter == "evidence_qa":
        return _convert_evidence_qa(row, metadata)
    if converter == "conversation":
        return _convert_conversation(row, metadata)
    if converter == "instruction_response":
        return _convert_instruction_response(row, metadata)
    raise ValueError(f"Unsupported converter: {converter}")


def build_public_sft(
    specs: list[DatasetSpec],
    *,
    selected_ids: set[str] | None,
    max_examples: int | None,
    streaming: bool,
) -> Iterator[dict[str, Any]]:
    selected = [spec for spec in specs if selected_ids is None or spec.id in selected_ids]
    if selected_ids:
        missing = selected_ids - {spec.id for spec in selected}
        if missing:
            raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(missing))}")

    for spec in selected:
        limit = max_examples if max_examples is not None else spec.default_max_examples
        emitted = 0
        splits = ("url",) if spec.url else spec.splits
        for split in splits:
            rows = iter_url_jsonl(spec.url) if spec.url else iter_hf_rows(spec, split=split, streaming=streaming)
            for row in rows:
                example = convert_row(
                    dict(row),
                    converter=spec.converter,
                    metadata={
                        "dataset_id": spec.id,
                        "hf_id": spec.hf_id,
                        "url": spec.url,
                        "split": split,
                        "purpose": spec.purpose,
                    },
                )
                if example is None:
                    continue
                yield example
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
            if limit is not None and emitted >= limit:
                break


def iter_hf_rows(spec: DatasetSpec, *, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'Hugging Face loading requires the optional dependency: python -m pip install -e ".[data]"'
        ) from exc

    kwargs: dict[str, Any] = {
        "path": spec.hf_id,
        "split": split,
        "streaming": streaming,
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.config:
        kwargs["name"] = spec.config
    try:
        return load_dataset(**kwargs)
    except TypeError as exc:
        if "Pickler._batch_setitems" in str(exc):
            raise RuntimeError(
                "This Hugging Face dataset loader path is incompatible with the current "
                "Python/dill combination. Use Python 3.11/3.12 for scripted HF datasets, "
                "or use URL/local JSONL datasets such as webgpt-comparisons."
            ) from exc
        raise


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            yield from _iter_jsonl_value(value, f"{path}:{line_number}")


def iter_url_jsonl(url: str | None) -> Iterator[dict[str, Any]]:
    if not url:
        raise ValueError("URL is required for URL JSONL loading")
    with urllib.request.urlopen(url, timeout=120) as response:
        for line_number, raw_line in enumerate(response, start=1):
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{url}:{line_number}: invalid JSONL") from exc
            yield from _iter_jsonl_value(value, f"{url}:{line_number}")


def _iter_jsonl_value(value: Any, location: str) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{location}[{index}]: expected JSON object")
            yield item
    else:
        raise ValueError(f"{location}: expected JSON object or array of objects")


def write_jsonl(examples: Iterable[dict[str, Any]], output: str | Path) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            validate_sft_example(example)
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
            count += 1
    return count


def validate_sft_example(example: dict[str, Any]) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT example requires at least two messages")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid message role: {message.get('role')}")
        if not str(message.get("content", "")).strip():
            raise ValueError("Message content cannot be empty")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Final SFT message must be assistant output")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build researcher-specialist SFT JSONL from public datasets."
    )
    parser.add_argument("--manifest", default="configs\\public_datasets.json")
    parser.add_argument("--dataset", action="append", help="Dataset id from manifest; repeatable")
    parser.add_argument("--output", required=True, help="Output chat SFT JSONL")
    parser.add_argument("--max-examples", type=int, help="Maximum examples per dataset")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--local-jsonl", help="Convert a local JSONL file instead of Hugging Face")
    parser.add_argument(
        "--converter",
        choices=["webgpt_comparisons", "evidence_qa", "conversation", "instruction_response"],
        help="Converter for --local-jsonl",
    )
    args = parser.parse_args(argv)

    if args.local_jsonl:
        if not args.converter:
            raise SystemExit("--converter is required with --local-jsonl")
        examples = (
            example
            for row in iter_jsonl(args.local_jsonl)
            for example in [convert_row(row, converter=args.converter, metadata={"dataset_id": "local"})]
            if example is not None
        )
    else:
        specs = load_manifest(args.manifest)
        selected_ids = set(args.dataset) if args.dataset else None
        examples = build_public_sft(
            specs,
            selected_ids=selected_ids,
            max_examples=args.max_examples,
            streaming=args.streaming,
        )

    count = write_jsonl(examples, args.output)
    print(f"Wrote {count} SFT examples to {args.output}")
    return 0


def _convert_webgpt_comparison(
    row: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    question = _first_text(row, ("question", "query", "prompt"))
    answer = _pick_webgpt_answer(row)
    if not question or not answer:
        return None
    sources = _extract_sources(row)
    return _make_example(
        question=question,
        answer=answer,
        task="Answer the research question using web evidence and citations.",
        sources=sources,
        metadata={**metadata, "converter": "webgpt_comparisons"},
    )


def _convert_evidence_qa(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    question = _first_text(row, ("question", "query", "claim", "input"))
    answer = _first_text(row, ("answer", "output", "final_answer", "label"))
    if not question or not answer:
        return None

    sources = _extract_sources(row)
    citations = _select_citations(row, sources)
    citation_text = " ".join(f"[{source_id}]" for source_id in citations)
    evidence_lines = [
        f"- [{source['id']}] {source['title']}: {source['text'][:500]}"
        for source in sources[:6]
    ]
    assistant = f"{answer}"
    if citation_text:
        assistant = f"{assistant} {citation_text}"
    if evidence_lines:
        assistant = f"{assistant}\n\nEvidence used:\n" + "\n".join(evidence_lines[:4])
    return _make_example(
        question=question,
        answer=assistant,
        task="Answer only from the provided evidence and cite source ids.",
        sources=sources,
        metadata={**metadata, "converter": "evidence_qa"},
    )


def _convert_conversation(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages") or row.get("conversations") or row.get("conversation")
    if not isinstance(messages, list):
        return _convert_instruction_response(row, metadata)

    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = _normalize_role(str(item.get("role") or item.get("from") or item.get("speaker") or ""))
        content = _textify(item.get("content") or item.get("value") or item.get("text"))
        if role and content:
            normalized.append({"role": role, "content": content})
    if not normalized or normalized[-1]["role"] != "assistant":
        return None
    if normalized[0]["role"] != "system":
        normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"messages": normalized, "metadata": {**metadata, "converter": "conversation"}}


def _convert_instruction_response(
    row: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    instruction = _first_text(row, ("instruction", "prompt", "question", "input", "query"))
    output = _first_text(row, ("output", "response", "answer", "completion", "final_answer"))
    if not instruction or not output:
        return None
    return _make_example(
        question=instruction,
        answer=output,
        task="Follow the instruction as a research-specialist assistant.",
        metadata={**metadata, "converter": "instruction_response"},
    )


def _make_example(
    *,
    question: str,
    answer: str,
    task: str,
    metadata: dict[str, Any],
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    user_payload: dict[str, Any] = {"task": task, "question": question}
    if sources:
        user_payload["sources"] = sources
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=True),
            },
            {"role": "assistant", "content": answer},
        ],
        "metadata": metadata,
    }


def _pick_webgpt_answer(row: dict[str, Any]) -> str:
    answer_0 = _textify(row.get("answer_0") or row.get("answer0") or row.get("response_0"))
    answer_1 = _textify(row.get("answer_1") or row.get("answer1") or row.get("response_1"))
    score_0 = _float_or_none(row.get("score_0") or row.get("score0"))
    score_1 = _float_or_none(row.get("score_1") or row.get("score1"))
    if answer_0 and answer_1 and score_0 is not None and score_1 is not None:
        return answer_0 if score_0 >= score_1 else answer_1
    return answer_0 or answer_1 or _first_text(row, ("answer", "response", "output"))


def _extract_sources(row: dict[str, Any]) -> list[dict[str, str]]:
    context = row.get("context") or row.get("contexts") or row.get("paragraphs")
    sources: list[dict[str, str]] = []
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("text") or context.get("texts") or []
        for index, title in enumerate(_as_list(titles), start=1):
            text = sentences[index - 1] if index - 1 < len(_as_list(sentences)) else ""
            sources.append(
                {
                    "id": f"S{index}",
                    "title": _textify(title) or f"Source {index}",
                    "text": _textify(text),
                }
            )
    elif isinstance(context, list):
        for index, item in enumerate(context, start=1):
            if isinstance(item, dict):
                title = _textify(item.get("title") or item.get("name")) or f"Source {index}"
                text = _textify(item.get("text") or item.get("sentences") or item.get("passage"))
            elif isinstance(item, (list, tuple)) and item:
                title = _textify(item[0]) or f"Source {index}"
                text = _textify(item[1:]) if len(item) > 1 else ""
            else:
                title = f"Source {index}"
                text = _textify(item)
            sources.append({"id": f"S{index}", "title": title, "text": text})

    for key in ("evidence", "supporting_context", "documents"):
        value = row.get(key)
        if value and not sources:
            for index, item in enumerate(_as_list(value), start=1):
                sources.append(
                    {
                        "id": f"S{index}",
                        "title": f"Evidence {index}",
                        "text": _textify(item),
                    }
                )
    quotes = row.get("quotes")
    if isinstance(quotes, list) and not sources:
        for index, quote in enumerate(quotes, start=1):
            if isinstance(quote, dict):
                title = _textify(quote.get("title")) or f"Quote {index}"
                text = _textify(quote.get("extract") or quote.get("text"))
            else:
                title = f"Quote {index}"
                text = _textify(quote)
            sources.append({"id": str(index), "title": title, "text": text})
    return [source for source in sources if source["text"]]


def _select_citations(row: dict[str, Any], sources: list[dict[str, str]]) -> list[str]:
    if not sources:
        return []
    supporting = row.get("supporting_facts") or row.get("supporting_facts_title")
    titles: set[str] = set()
    if isinstance(supporting, dict):
        titles.update(_textify(title) for title in _as_list(supporting.get("title")))
    elif isinstance(supporting, list):
        for item in supporting:
            if isinstance(item, (list, tuple)) and item:
                titles.add(_textify(item[0]))
            elif isinstance(item, dict):
                titles.add(_textify(item.get("title") or item.get("name")))
    citations = [
        source["id"]
        for source in sources
        if not titles or source["title"] in titles
    ]
    return citations[:3] if citations else [sources[0]["id"]]


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _textify(row.get(key))
        if text:
            return text
    return ""


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, dict):
        for key in ("full_text", "text", "answer", "content", "value", "response"):
            text = _textify(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        return " ".join(_textify(item) for item in value if _textify(item)).strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_role(role: str) -> str:
    lowered = role.lower()
    if lowered in {"system"}:
        return "system"
    if lowered in {"user", "human", "customer"}:
        return "user"
    if lowered in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    if lowered in {"tool", "function"}:
        return "tool"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
