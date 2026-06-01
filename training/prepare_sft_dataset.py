from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL trace line") from exc
    return events


def trace_to_sft_example(events: list[dict[str, Any]]) -> dict[str, Any]:
    question = _first_payload(events, "question").get("question", "")
    plan = _first_payload(events, "plan")
    notes = _last_payload(events, "notes", default=[])
    final = _first_payload(events, "final_answer").get("markdown", "")
    if not question or not final:
        raise ValueError("Trace must contain question and final_answer events")
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a local DeepResearch agent. Plan, use tools, verify evidence, "
                    "and answer with citations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "plan": plan, "notes": notes},
                    ensure_ascii=True,
                ),
            },
            {"role": "assistant", "content": final},
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL from research traces.")
    parser.add_argument("traces", nargs="+", help="Trace JSONL files or directories")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    trace_paths = list(_expand_paths([Path(item) for item in args.traces]))
    if not trace_paths:
        raise SystemExit("No trace files found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trace_path in trace_paths:
            example = trace_to_sft_example(load_events(trace_path))
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    return 0


def _expand_paths(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.jsonl"))
        elif path.is_file():
            yield path


def _first_payload(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return _last_payload(events, name, default={})


def _last_payload(events: list[dict[str, Any]], name: str, *, default: Any) -> Any:
    payload = default
    for event in events:
        if event.get("event") == name:
            payload = event.get("payload", default)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
