from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Config
from .json_utils import repair_json
from .llm import ChatClient, LLMError, create_client
from .models import Note, PlanStep, ResearchPlan, ResearchReport, SearchResult, SourceDocument
from .notes import extract_keywords, extract_notes
from .readers import ReadError, WebReader
from .report import ReportError, write_deterministic_report, write_llm_report
from .repository import RepositoryAnalyzer
from .search import SearchError, Searcher, create_searcher
from .tracing import TraceCollector
from .verification import validate_report_citations, verify_evidence


class AgentError(RuntimeError):
    """Raised when the research workflow cannot continue."""


class ResearchAgent:
    def __init__(
        self,
        config: Config,
        *,
        client: ChatClient | None = None,
        searcher: Searcher | None = None,
        reader: WebReader | None = None,
    ) -> None:
        self.config = config
        self.client = client or create_client(config.model)
        self.searcher = searcher or create_searcher(config.search)
        self.reader = reader or WebReader(config.search)

    def research(
        self,
        question: str,
        *,
        repo_path: str | None = None,
        trace: bool = True,
    ) -> ResearchReport:
        if not question.strip():
            raise AgentError("Question cannot be empty")

        tracer = TraceCollector(self.config.agent.trace_dir, question, enabled=trace)
        tracer.add("question", {"question": question, "repo_path": repo_path})
        if repo_path:
            return self._research_repository(question, repo_path, tracer)
        return self._research_web(question, tracer)

    def _research_web(self, question: str, tracer: TraceCollector) -> ResearchReport:
        plan = self._create_plan(question)
        tracer.add("plan", plan)

        all_results: list[SearchResult] = []
        all_sources: list[SourceDocument] = []
        all_notes: list[Note] = []
        seen_urls: set[str] = set()
        verification = verify_evidence(
            question, all_notes, all_sources, min_sources=self.config.agent.min_sources
        )

        queries = list(plan.queries)
        for iteration in range(1, self.config.agent.max_iterations + 1):
            tracer.add("iteration_start", {"iteration": iteration, "queries": queries})
            for query in queries:
                tracer.add("tool_call", {"tool": "search", "query": query})
                try:
                    results = self.searcher.search(query)
                except SearchError as exc:
                    tracer.add("tool_error", {"tool": "search", "query": query, "error": str(exc)})
                    raise AgentError(str(exc)) from exc
                tracer.add("search_results", results)
                all_results.extend(results)

            for result in _dedupe_results(all_results):
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                tracer.add("tool_call", {"tool": "read", "url": result.url})
                try:
                    source = self.reader.read(result)
                except ReadError as exc:
                    tracer.add("tool_error", {"tool": "read", "url": result.url, "error": str(exc)})
                    continue
                all_sources.append(source)
                tracer.add("source", source)

            all_notes = extract_notes(
                question, all_sources, max_notes=self.config.agent.max_notes
            )
            tracer.add("notes", all_notes)
            verification = verify_evidence(
                question,
                all_notes,
                all_sources,
                min_sources=self.config.agent.min_sources,
            )
            tracer.add("verification", verification)
            if verification.enough_evidence:
                break
            queries = _gap_queries(question, verification.gaps)

        markdown = self._write_report(question, plan, all_notes, all_sources, verification)
        citation_check = validate_report_citations(markdown, all_sources)
        if all_sources and not citation_check.ok:
            messages = "; ".join(issue.message for issue in citation_check.issues)
            raise AgentError(f"Report citation validation failed: {messages}")

        tracer.add("final_answer", {"markdown": markdown})
        return ResearchReport(
            markdown=markdown,
            sources=all_sources,
            notes=all_notes,
            verification=verification,
            trace_path=str(tracer.path) if tracer.path else None,
        )

    def _research_repository(
        self, question: str, repo_path: str, tracer: TraceCollector
    ) -> ResearchReport:
        analyzer = RepositoryAnalyzer(self.config.repository)
        index = analyzer.index(repo_path)
        tracer.add("repository_index", {"root": str(Path(repo_path).resolve()), "files": len(index.files)})
        markdown, sources = analyzer.report(index, question)
        verification = validate_report_citations(markdown, sources)
        if not verification.ok:
            messages = "; ".join(issue.message for issue in verification.issues)
            raise AgentError(f"Repository report citation validation failed: {messages}")
        tracer.add("final_answer", {"markdown": markdown})
        return ResearchReport(
            markdown=markdown,
            sources=sources,
            notes=[],
            verification=verification,
            trace_path=str(tracer.path) if tracer.path else None,
        )

    def _create_plan(self, question: str) -> ResearchPlan:
        if self.client.provider == "offline":
            return _deterministic_plan(question)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the planning component of a local research agent. Return "
                    "strict JSON only, with no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a compact research plan and search queries for this question. "
                    "Return JSON with shape: "
                    "{\"steps\":[{\"id\":\"1\",\"goal\":\"...\"}],\"queries\":[\"...\"]}. "
                    f"Question: {question}"
                ),
            },
        ]
        try:
            raw = self.client.complete(messages)
            data = repair_json(raw)
        except (LLMError, ValueError) as exc:
            raise AgentError(f"Planning failed: {exc}") from exc

        try:
            steps = [
                PlanStep(id=str(item["id"]), goal=str(item["goal"]))
                for item in data["steps"]
            ]
            queries = [str(query) for query in data["queries"]]
        except (KeyError, TypeError) as exc:
            raise AgentError("Planning JSON missing required steps or queries") from exc
        if not steps or not queries:
            raise AgentError("Planning produced no steps or queries")
        return ResearchPlan(question=question, steps=steps, queries=queries[:5])

    def _write_report(
        self,
        question: str,
        plan: ResearchPlan,
        notes: list[Note],
        sources: list[SourceDocument],
        verification,
    ) -> str:
        if self.client.provider == "offline" or not self.config.agent.use_llm_report:
            return write_deterministic_report(question, plan, notes, sources, verification)
        try:
            return write_llm_report(self.client, question, plan, notes, sources, verification)
        except ReportError:
            raise
        except LLMError as exc:
            raise AgentError(f"Report writing failed: {exc}") from exc


def _deterministic_plan(question: str) -> ResearchPlan:
    keywords = extract_keywords(question, limit=6)
    core_query = " ".join(keywords) if keywords else question
    steps = [
        PlanStep(id="1", goal="Clarify the research target and expected evidence."),
        PlanStep(id="2", goal="Search for primary and explanatory sources."),
        PlanStep(id="3", goal="Extract source-backed notes and identify gaps."),
        PlanStep(id="4", goal="Write a cited answer or state insufficient evidence."),
    ]
    queries = [question]
    if core_query and core_query.lower() != question.lower():
        queries.append(core_query)
    return ResearchPlan(question=question, steps=steps, queries=queries[:3])


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        deduped.append(result)
    return deduped


def _gap_queries(question: str, gaps: list[str]) -> list[str]:
    if not gaps:
        return [question]
    return [f"{question} {gap}"[:180] for gap in gaps[:3]]
