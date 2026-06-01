from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\\/\-]+)\]")


@dataclass
class DatasetIssue:
    line: int
    code: str
    message: str


@dataclass
class DatasetStats:
    examples: int = 0
    assistant_chars: int = 0
    examples_with_citations: int = 0
    total_citations: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def avg_assistant_chars(self) -> float:
        return self.assistant_chars / self.examples if self.examples else 0.0

    @property
    def issue_count(self) -> int:
        return len(self.issues)


def validate_dataset(path: str | Path) -> DatasetStats:
    stats = DatasetStats()
    for line_number, example in read_jsonl(path):
        stats.examples += 1
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            stats.issues.append(
                DatasetIssue(line_number, "bad_messages", "Expected at least three chat messages.")
            )
            continue

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                stats.issues.append(
                    DatasetIssue(line_number, "bad_message", f"Message {index} is not an object.")
                )
                continue
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                stats.issues.append(
                    DatasetIssue(
                        line_number,
                        "bad_role",
                        f"Message {index} has invalid role {message.get('role')!r}.",
                    )
                )
            if not str(message.get("content", "")).strip():
                stats.issues.append(
                    DatasetIssue(line_number, "empty_content", f"Message {index} is empty.")
                )

        final = messages[-1] if isinstance(messages[-1], dict) else {}
        if final.get("role") != "assistant":
            stats.issues.append(
                DatasetIssue(line_number, "bad_final_role", "Final message must be assistant.")
            )
            continue

        assistant = str(final.get("content", "")).strip()
        stats.assistant_chars += len(assistant)
        citations = _CITATION_RE.findall(assistant)
        stats.total_citations += len(citations)
        if citations:
            stats.examples_with_citations += 1
        elif _has_sources(messages):
            stats.issues.append(
                DatasetIssue(line_number, "missing_citation", "Example has sources but no citations.")
            )

        allowed = _source_ids_from_user(messages)
        invalid = sorted(set(citations) - allowed) if allowed else []
        if invalid:
            stats.issues.append(
                DatasetIssue(
                    line_number,
                    "invalid_citation",
                    f"Unknown citation id(s): {', '.join(invalid)}.",
                )
            )
    return stats


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


def render_stats(stats: DatasetStats) -> str:
    lines = [
        f"examples: {stats.examples}",
        f"avg_assistant_chars: {stats.avg_assistant_chars:.1f}",
        f"examples_with_citations: {stats.examples_with_citations}",
        f"total_citations: {stats.total_citations}",
        f"issues: {stats.issue_count}",
    ]
    for issue in stats.issues[:50]:
        lines.append(f"- line {issue.line}: {issue.code}: {issue.message}")
    if len(stats.issues) > 50:
        lines.append(f"- ... {len(stats.issues) - 50} more issue(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate chat SFT JSONL quality.")
    parser.add_argument("dataset", help="Chat SFT JSONL path")
    parser.add_argument("--json", action="store_true", help="Print JSON stats")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when issues are found")
    args = parser.parse_args(argv)

    stats = validate_dataset(args.dataset)
    if args.json:
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=True))
    else:
        print(render_stats(stats))
    return 1 if args.strict and stats.issues else 0


def _has_sources(messages: list[dict[str, Any]]) -> bool:
    return bool(_source_ids_from_user(messages))


def _source_ids_from_user(messages: list[dict[str, Any]]) -> set[str]:
    source_ids: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for source in payload.get("sources", []) if isinstance(payload, dict) else []:
            if isinstance(source, dict) and source.get("id") is not None:
                source_ids.add(str(source["id"]))
    return source_ids


if __name__ == "__main__":
    raise SystemExit(main())
