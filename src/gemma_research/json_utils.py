from __future__ import annotations

import json
import re
from typing import Any


class JSONRepairError(ValueError):
    """Raised when model output cannot be repaired into valid JSON."""


def repair_json(text: str) -> Any:
    candidate = _strip_fences(text.strip())
    candidate = _extract_balanced_json(candidate)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = candidate[:500].replace("\n", "\\n")
        raise JSONRepairError(f"Invalid JSON after repair: {exc}; preview={preview}") from exc


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _extract_balanced_json(text: str) -> str:
    start = min(
        (index for index in (text.find("{"), text.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        raise JSONRepairError("No JSON object or array found in model output")

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise JSONRepairError("JSON object or array is not balanced")
