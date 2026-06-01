# Agent Flow

## Web research mode

1. The CLI loads TOML config and applies command-line overrides.
2. `ResearchAgent` creates a trace file unless tracing is disabled.
3. The planner returns a `ResearchPlan` with steps and search queries.
4. Each iteration runs search queries and deduplicates URLs.
5. The reader fetches web pages and converts HTML into plain text.
6. The note extractor selects source-backed claims relevant to the question.
7. The verifier checks minimum source count and uncovered query terms.
8. If evidence is sufficient, the report writer generates the final markdown.
9. If gaps remain and the iteration limit has not been reached, gap-oriented queries are generated and the loop repeats.
10. Citation validation checks that every bracketed citation refers to a known source ID.

## Repository mode

`gemma-research --repo . "Analyze this codebase"` switches to read-only repository analysis. The analyzer indexes text files, ignores common generated directories, searches relevant files, extracts dependency signals, scans common risk patterns, and writes a cited repository report using file paths as citation IDs.

## Termination

The loop terminates when one of these conditions is true:

- verification says enough evidence was collected;
- `agent.max_iterations` is reached;
- a fatal configuration, search, read, or model error occurs.

When evidence remains insufficient, the report explicitly says so instead of fabricating an answer.
