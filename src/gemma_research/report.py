from __future__ import annotations

from .llm import ChatClient, LLMError
from .models import Note, ResearchPlan, SourceDocument, VerificationResult
from .verification import validate_report_citations


class ReportError(RuntimeError):
    """Raised when a report cannot be generated safely."""


def write_deterministic_report(
    question: str,
    plan: ResearchPlan,
    notes: list[Note],
    sources: list[SourceDocument],
    verification: VerificationResult,
) -> str:
    lines = [f"# Research Report", "", f"**Question:** {question}", ""]
    if not verification.enough_evidence:
        lines.extend(
            [
                "## Evidence Status",
                "Insufficient evidence was collected to fully answer the question.",
                "",
            ]
        )

    lines.extend(["## Method", "The agent followed this plan:", ""])
    for step in plan.steps:
        lines.append(f"{step.id}. {step.goal}")
    lines.append("")

    if notes:
        lines.extend(["## Findings", ""])
        for index, note in enumerate(notes[:12], start=1):
            lines.append(f"{index}. {note.claim} [{note.source_id}]")
        lines.append("")
    else:
        lines.extend(
            [
                "## Findings",
                "No source-backed findings were extracted. Run with a web search provider or repository input.",
                "",
            ]
        )

    if verification.gaps:
        lines.extend(["## Gaps and Verification", ""])
        for gap in verification.gaps:
            lines.append(f"- {gap}")
        lines.append("")

    if sources:
        lines.extend(["## Sources", ""])
        for source in sources:
            locator = source.path or source.url
            lines.append(f"- [{source.id}] {source.title} - {locator}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_llm_report(
    client: ChatClient,
    question: str,
    plan: ResearchPlan,
    notes: list[Note],
    sources: list[SourceDocument],
    verification: VerificationResult,
) -> str:
    if not notes:
        return write_deterministic_report(question, plan, notes, sources, verification)

    note_block = "\n".join(
        f"- [{note.source_id}] {note.claim}\n  Quote: {note.quote}\n  Source: {note.title} {note.url}"
        for note in notes
    )
    source_ids = ", ".join(source.id for source in sources)
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise research reports grounded only in supplied notes. "
                "Every factual claim must cite one of the provided source ids in square "
                "brackets. Do not cite unknown ids. If evidence is insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Allowed citations: {source_ids}\n"
                f"Evidence status: {'sufficient' if verification.enough_evidence else 'insufficient'}\n\n"
                f"Notes:\n{note_block}\n\n"
                "Write a markdown report with sections: Answer, Key Findings, Gaps, Sources."
            ),
        },
    ]
    try:
        markdown = client.complete(messages)
    except LLMError:
        raise
    citation_check = validate_report_citations(markdown, sources)
    if not citation_check.ok:
        messages = "; ".join(issue.message for issue in citation_check.issues)
        raise ReportError(f"Model report failed citation validation: {messages}")
    return markdown.strip() + "\n"
