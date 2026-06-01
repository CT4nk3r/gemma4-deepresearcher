from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceCollector:
    def __init__(self, trace_dir: str | Path, question: str, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path: Path | None = None
        if enabled:
            root = Path(trace_dir)
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            slug = _slug(question)
            self.path = root / f"{stamp}-{slug}.jsonl"

    def add(self, event_type: str, payload: Any) -> None:
        if not self.enabled or self.path is None:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": _jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return (slug or "research")[:60]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
