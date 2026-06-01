from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from training.validate_sft_dataset import validate_dataset
except ModuleNotFoundError:
    from validate_sft_dataset import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a clean SFT JSONL by dropping invalid rows.")
    parser.add_argument("--input", required=True, help="Input chat SFT JSONL")
    parser.add_argument("--output", required=True, help="Clean output chat SFT JSONL")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate user prompts instead of dropping later duplicates",
    )
    args = parser.parse_args(argv)

    bad_lines = {issue.line for issue in validate_dataset(args.input).issues}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped_invalid = 0
    dropped_duplicate = 0
    seen_prompts: set[str] = set()
    with Path(args.input).open("r", encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if line_number in bad_lines:
                dropped_invalid += 1
                continue
            example = _parse_json_object(line, args.input, line_number)
            fingerprint = _prompt_fingerprint(example)
            if not args.no_dedupe and fingerprint in seen_prompts:
                dropped_duplicate += 1
                continue
            seen_prompts.add(fingerprint)
            target.write(json.dumps(example, ensure_ascii=True) + "\n")
            kept += 1
    print(f"kept: {kept}")
    print(f"dropped_invalid: {dropped_invalid}")
    print(f"dropped_duplicate: {dropped_duplicate}")
    print(f"output: {output}")
    return 0


def _parse_json_object(line: str, path: str, line_number: int) -> dict[str, Any]:
    value: Any = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: expected JSON object")
    return value


def _prompt_fingerprint(example: dict[str, Any]) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return json.dumps(example, sort_keys=True, ensure_ascii=True)
    user_parts = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return "\n".join(user_parts)


if __name__ == "__main__":
    raise SystemExit(main())
