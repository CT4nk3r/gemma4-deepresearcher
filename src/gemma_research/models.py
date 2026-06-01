from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    id: str
    goal: str
    status: str = "pending"


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    steps: list[PlanStep]
    queries: list[str]


@dataclass(frozen=True)
class SearchResult:
    query: str
    title: str
    url: str
    snippet: str
    rank: int


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    url: str
    text: str
    source_type: str = "web"
    path: str | None = None


@dataclass(frozen=True)
class Note:
    source_id: str
    title: str
    url: str
    claim: str
    quote: str
    relevance: str


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    enough_evidence: bool
    issues: list[VerificationIssue] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    cited_source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchReport:
    markdown: str
    sources: list[SourceDocument]
    notes: list[Note]
    verification: VerificationResult
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
