from __future__ import annotations

import html
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from .config import SearchConfig
from .models import SearchResult


class SearchError(RuntimeError):
    """Raised when search fails."""


class Searcher(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        """Return ranked search results for a query."""


@dataclass
class NoSearch:
    def search(self, query: str) -> list[SearchResult]:
        return []


@dataclass
class DuckDuckGoSearcher:
    config: SearchConfig

    def search(self, query: str) -> list[SearchResult]:
        if not query.strip():
            raise SearchError("Search query cannot be empty")
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.config.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SearchError(f"DuckDuckGo search failed for query '{query}': {exc}") from exc

        parser = _DuckDuckGoParser(query=query, max_results=self.config.max_results)
        parser.feed(body)
        return parser.results


class _DuckDuckGoParser(HTMLParser):
    def __init__(self, *, query: str, max_results: int) -> None:
        super().__init__()
        self.query = query
        self.max_results = max_results
        self.results: list[SearchResult] = []
        self._in_result_link = False
        self._current_href = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or len(self.results) >= self.max_results:
            return
        attr_map = {name: value or "" for name, value in attrs}
        css_class = attr_map.get("class", "")
        if "result__a" in css_class:
            self._in_result_link = True
            self._current_href = attr_map.get("href", "")
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_link:
            title = html.unescape(" ".join(self._current_text)).strip()
            url = _decode_duckduckgo_url(self._current_href)
            if title and url:
                rank = len(self.results) + 1
                self.results.append(
                    SearchResult(
                        query=self.query,
                        title=title,
                        url=url,
                        snippet="",
                        rank=rank,
                    )
                )
            self._in_result_link = False
            self._current_href = ""
            self._current_text = []


def _decode_duckduckgo_url(raw_url: str) -> str:
    url = html.unescape(raw_url)
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        url = "https://duckduckgo.com" + url
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def create_searcher(config: SearchConfig) -> Searcher:
    provider = config.provider.lower()
    if provider == "duckduckgo":
        return DuckDuckGoSearcher(config)
    if provider == "none":
        return NoSearch()
    raise SearchError(f"Unsupported search provider: {config.provider}")
