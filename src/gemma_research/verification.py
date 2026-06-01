from __future__ import annotations

import re

from .models import Note, SourceDocument, VerificationIssue, VerificationResult
from .notes import extract_keywords


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\\/\-]+)\]")


def verify_evidence(
    question: str,
    notes: list[Note],
    sources: list[SourceDocument],
    *,
    min_sources: int,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    gaps: list[str] = []
    note_source_ids = {note.source_id for note in notes}

    if len(note_source_ids) < min_sources:
        gaps.append(
            f"Only {len(note_source_ids)} source(s) produced relevant notes; "
            f"{min_sources} required."
        )
        issues.append(
            VerificationIssue(
                code="insufficient_sources",
                message="Not enough independent sources support the answer.",
                severity="error",
            )
        )

    keywords = extract_keywords(question, limit=8)
    combined_notes = " ".join(note.claim for note in notes).lower()
    missing_keywords = [keyword for keyword in keywords if keyword not in combined_notes]
    if missing_keywords:
        gaps.append(f"Evidence does not directly cover: {', '.join(missing_keywords[:5])}.")

    source_ids = {source.id for source in sources}
    orphan_notes = sorted(note_source_ids - source_ids)
    if orphan_notes:
        issues.append(
            VerificationIssue(
                code="orphan_notes",
                message=f"Notes reference unknown source ids: {', '.join(orphan_notes)}.",
                severity="error",
            )
        )

    enough_evidence = not any(issue.severity == "error" for issue in issues)
    return VerificationResult(
        ok=enough_evidence,
        enough_evidence=enough_evidence,
        issues=issues,
        gaps=gaps,
        cited_source_ids=sorted(note_source_ids),
    )


def validate_report_citations(
    markdown: str, sources: list[SourceDocument]
) -> VerificationResult:
    source_ids = {source.id for source in sources}
    path_ids = {source.path for source in sources if source.path}
    allowed = source_ids | {path for path in path_ids if path}
    cited = sorted(set(_CITATION_RE.findall(markdown)))
    issues: list[VerificationIssue] = []
    for citation in cited:
        if citation not in allowed:
            issues.append(
                VerificationIssue(
                    code="invalid_citation",
                    message=f"Report cites unknown source [{citation}].",
                    severity="error",
                )
            )

    if sources and not cited:
        issues.append(
            VerificationIssue(
                code="missing_citations",
                message="Report contains sources but no citations.",
                severity="error",
            )
        )

    ok = not any(issue.severity == "error" for issue in issues)
    return VerificationResult(
        ok=ok,
        enough_evidence=ok,
        issues=issues,
        gaps=[],
        cited_source_ids=cited,
    )
