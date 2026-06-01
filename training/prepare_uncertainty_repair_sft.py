from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a local DeepResearch agent. Use only the provided sources, cite source ids "
    "in square brackets, separate evidence from interpretation, and explicitly say when "
    "the evidence is insufficient."
)


REPO_CASES = [
    {
        "topic": "whether a cache change fixed API latency",
        "claim": "The new cache layer fixed the API latency regression.",
        "sources": [
            ("S1", "Commit message", "Add in-memory cache for repository metadata lookups. No benchmark results are included."),
            ("S2", "Issue comment", "One maintainer says the endpoint feels faster locally after the change."),
            ("S3", "CI log", "Unit tests pass. No load test or production latency dashboard is attached."),
        ],
        "missing": "before/after latency metrics from comparable traffic, production monitoring, or load tests",
    },
    {
        "topic": "whether a refactor caused failing Windows tests",
        "claim": "The parser refactor caused the Windows-only test failure.",
        "sources": [
            ("S1", "Pull request summary", "Refactors path normalization and removes duplicate parsing helpers."),
            ("S2", "CI summary", "Windows tests failed in test_path_roundtrip. Linux and macOS tests passed."),
            ("S3", "Recent issue", "A separate dependency update changed pathlib behavior two hours before the PR run."),
        ],
        "missing": "bisect results, a rerun against the old dependency, or a minimal reproduction",
    },
    {
        "topic": "whether adding retries improved deployment reliability",
        "claim": "The retry wrapper made deployments reliable.",
        "sources": [
            ("S1", "Release note", "Adds retry logic around transient HTTP 502 responses during deployment."),
            ("S2", "Incident report", "The previous outage involved DNS propagation delays, not HTTP 502 responses."),
            ("S3", "Dashboard excerpt", "Deployment success rate is not shown for the period after the release."),
        ],
        "missing": "post-release success rates and evidence that the prior failure mode matches the retry condition",
    },
    {
        "topic": "whether a security patch closed the reported vulnerability",
        "claim": "The patch fully fixes the reported authentication bypass.",
        "sources": [
            ("S1", "Security advisory", "The bypass occurs when a missing tenant id falls back to a default tenant."),
            ("S2", "Patch diff summary", "Adds a tenant id null check in the REST API handler."),
            ("S3", "Code search note", "A GraphQL resolver still accepts requests before tenant id validation."),
        ],
        "missing": "coverage of all affected entry points and a regression test for the advisory scenario",
    },
    {
        "topic": "whether a memory leak is solved",
        "claim": "The memory leak is solved by closing database cursors.",
        "sources": [
            ("S1", "Bug report", "Memory grows during long websocket sessions while database traffic is idle."),
            ("S2", "Patch summary", "Closes database cursors in a batch import path."),
            ("S3", "Profiler note", "No heap snapshot is attached after applying the patch."),
        ],
        "missing": "heap profiles or reproduction results that connect cursor lifetime to the websocket leak",
    },
]


CAUSAL_CASES = [
    (
        "a city minimum wage increase caused restaurant closures",
        "restaurant closures rose after the wage increase",
        "local rents and food costs also rose during the same period",
        "matched comparison cities or firm-level closure data controlling for other cost shocks",
    ),
    (
        "a school phone ban improved student attention",
        "teachers reported fewer visible phone distractions",
        "the district also shortened class periods and changed attendance enforcement",
        "pre/post attention measures with a comparison group or randomized rollout",
    ),
    (
        "an AI coding assistant reduced production bugs by 40 percent",
        "the company reported fewer bug tickets after rollout",
        "the team simultaneously changed release gates and QA staffing",
        "a controlled measurement design separating the assistant from process changes",
    ),
    (
        "a new policing program reduced local crime",
        "reported crime fell after the program launched",
        "neighboring areas also saw declines during the same season",
        "baseline trends, comparison areas, and reporting-rate checks",
    ),
    (
        "a tutoring intervention raised test scores",
        "participants scored higher at year end",
        "participants volunteered and had higher prior attendance",
        "random assignment or matched controls for motivation and prior achievement",
    ),
]


SOURCE_MISMATCH_CASES = [
    (
        "whether a startup's desalination process is net-zero",
        "the plant uses renewable electricity",
        "the source does not discuss embodied emissions, brine handling, or backup power",
        "lifecycle emissions across construction, operation, maintenance, and brine disposal",
    ),
    (
        "whether a battery chemistry will make EVs cheaper within two years",
        "lab energy density improved in coin-cell tests",
        "the source does not include manufacturing yield, supply cost, or automotive qualification timelines",
        "scale-up cost data and production validation timelines",
    ),
    (
        "whether a supplement improves sleep quality",
        "a small uncontrolled survey reports better sleep",
        "the source lacks randomization, blinding, and objective sleep measurement",
        "randomized controlled evidence with validated sleep outcomes",
    ),
    (
        "whether open-source software is inherently more secure",
        "a report says public review can help find vulnerabilities",
        "the source also notes maintainer capacity and patch latency vary widely",
        "comparative vulnerability and remediation data across similar projects",
    ),
    (
        "whether remote work improves engineering productivity",
        "employees report higher satisfaction",
        "the source does not measure shipped work, quality, or cycle time",
        "objective productivity and quality metrics with role and team controls",
    ),
]


CONFLICT_CASES = [
    (
        "which of two medical studies deserves more weight",
        "Study A is a randomized trial with 1,200 patients and a prespecified endpoint.",
        "Study B is a retrospective cohort with 80 patients and incomplete follow-up.",
        "study design, sample size, bias risk, endpoint relevance, and consistency with other evidence",
    ),
    (
        "how to handle sources that agree on measurements but disagree on interpretation",
        "Both reports show the same temperature trend over ten years.",
        "One report attributes the change to policy, while the other says attribution was not tested.",
        "a clear separation between shared measurements and unsupported causal interpretation",
    ),
    (
        "whether a tiny statistically significant study is practically meaningful",
        "The study reports p < 0.05 in 18 participants.",
        "The measured effect is smaller than the stated minimum clinically important difference.",
        "confidence intervals, effect size, replication, and practical significance",
    ),
    (
        "whether outdated evidence applies to current GPU buying advice",
        "A 2023 benchmark ranks several GPUs for local inference.",
        "The question asks for current fine-tuning value, but prices and drivers have changed.",
        "current prices, VRAM requirements, driver support, and workload-specific benchmarks",
    ),
    (
        "whether cancer incidence increased because of screening",
        "Registries show more diagnoses after a screening campaign.",
        "Mortality is unchanged and coding rules changed during the same interval.",
        "stage distribution, screening rates, diagnostic criteria, mortality, and artifact checks",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build targeted uncertainty-repair SFT seeds.")
    parser.add_argument("--output", default="data\\uncertainty_repair_seed_sft.jsonl")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat template families with prompt variants")
    args = parser.parse_args(argv)

    examples = list(build_examples(repeat=args.repeat))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    print(f"Wrote {len(examples)} uncertainty repair seed examples to {output}")
    return 0


def build_examples(*, repeat: int = 2) -> Iterable[dict[str, Any]]:
    for variant in range(repeat):
        for case in REPO_CASES:
            yield make_example(
                question=repo_question(case["topic"], case["claim"], variant),
                sources=case["sources"],
                missing=case["missing"],
                bad_answer=f"{case['claim']} The available repository evidence is enough to accept the claim.",
                category="repo_uncertainty",
            )
        for claim, observed, confounder, missing in CAUSAL_CASES:
            yield make_example(
                question=causal_question(claim, variant),
                sources=[
                    ("S1", "Observed outcome", observed + "."),
                    ("S2", "Alternative explanation", confounder + "."),
                    ("S3", "Evidence gap", "No randomized design, matched comparison, or causal model is provided."),
                ],
                missing=missing,
                bad_answer=f"The claim that {claim} is supported because {observed}.",
                category="causal_uncertainty",
            )
        for topic, supported, gap, missing in SOURCE_MISMATCH_CASES:
            yield make_example(
                question=mismatch_question(topic, variant),
                sources=[
                    ("S1", "Supported fact", supported + "."),
                    ("S2", "Scope limitation", gap + "."),
                ],
                missing=missing,
                bad_answer=f"The available source supports the broader claim about {topic}.",
                category="source_mismatch",
            )
        for topic, source_a, source_b, missing in CONFLICT_CASES:
            yield make_example(
                question=conflict_question(topic, variant),
                sources=[
                    ("S1", "Source A", source_a),
                    ("S2", "Source B", source_b),
                ],
                missing=missing,
                bad_answer="The answer is straightforward because both sources can be combined into one conclusion.",
                category="conflict_or_scope",
            )


def repo_question(topic: str, claim: str, variant: int) -> str:
    if variant % 2 == 0:
        return f'A repository discussion claims: "{claim}" How should a careful repo research answer evaluate {topic}?'
    return f"What can and cannot be concluded from the provided repo evidence about {topic}?"


def causal_question(claim: str, variant: int) -> str:
    if variant % 2 == 0:
        return f"Assess the claim that {claim}. What would the evidence need to show?"
    return f"What should a research answer say before accepting that {claim}?"


def mismatch_question(topic: str, variant: int) -> str:
    if variant % 2 == 0:
        return f"Evaluate {topic} using only the provided sources."
    return f"How should an answer avoid overclaiming when asked about {topic}?"


def conflict_question(topic: str, variant: int) -> str:
    if variant % 2 == 0:
        return f"Give a careful research answer about {topic}."
    return f"What uncertainty or qualification is needed when answering {topic}?"


def make_example(
    *,
    question: str,
    sources: list[tuple[str, str, str]],
    missing: str,
    bad_answer: str,
    category: str,
) -> dict[str, Any]:
    payload = {
        "task": (
            "Answer the research question using only the provided sources. "
            "If the evidence is insufficient, say exactly what is missing."
        ),
        "question": question,
        "sources": [
            {"id": source_id, "title": title, "text": text}
            for source_id, title, text in sources
        ],
        "expected_behavior": [
            "state insufficient evidence when warranted",
            "cite every factual claim",
            "separate evidence from interpretation",
            "name missing evidence",
        ],
        "missing_evidence_hint": missing,
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            {"role": "assistant", "content": bad_answer},
        ],
        "metadata": {
            "dataset_id": "uncertainty-repair-seed",
            "purpose": "Targeted repair examples for evidence insufficiency and cautious repository research.",
            "category": category,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
