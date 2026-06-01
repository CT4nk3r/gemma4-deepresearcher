from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from .config import SearchConfig
from .models import SearchResult, SourceDocument


class ReadError(RuntimeError):
    """Raised when a source cannot be fetched or parsed."""


@dataclass
class WebReader:
    config: SearchConfig
    max_chars: int = 24_000

    def read(self, result: SearchResult) -> SourceDocument:
        request = urllib.request.Request(
            result.url,
            headers={"User-Agent": self.config.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read(self.max_chars * 4)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ReadError(f"Failed to read source {result.url}: {exc}") from exc

        if "text" not in content_type and "html" not in content_type and content_type:
            raise ReadError(f"Unsupported content type for {result.url}: {content_type}")

        text_body = body.decode(_encoding_from_content_type(content_type), errors="replace")
        parser = _ReadableHTMLParser()
        parser.feed(text_body)
        title = parser.title or result.title
        text = parser.text()
        if not text:
            text = re.sub(r"\s+", " ", text_body).strip()
        return SourceDocument(
            id=f"S{result.rank}",
            title=title[:200],
            url=result.url,
            text=text[: self.max_chars],
            source_type="web",
        )


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "article", "section"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = data.strip()
        if not clean:
            return
        if self._in_title:
            self.title = f"{self.title} {clean}".strip()
        else:
            self._parts.append(clean)

    def text(self) -> str:
        joined = " ".join(self._parts)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n\s+", "\n", joined)
        return joined.strip()


def _encoding_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, flags=re.IGNORECASE)
    return match.group(1) if match else "utf-8"
