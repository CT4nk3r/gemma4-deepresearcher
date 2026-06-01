# Tools

## Model providers

- `lmstudio`: OpenAI-compatible `/v1/chat/completions`, default `http://localhost:1234/v1`.
- `ollama`: native `/api/chat`, default `http://localhost:11434`.
- `openai-compatible`: any compatible endpoint with explicit `model.base_url`.
- `offline`: deterministic local provider for tests and smoke checks.

## Search

- `duckduckgo`: HTML search parser using the configured user agent and timeout.
- `none`: disables web search, useful for offline development.

Alibaba-NLP/DeepResearch uses Serper for Google web/scholar search and Jina Reader for web extraction. This project keeps those as replaceable interfaces and defaults to dependency-free Python implementations.

## Reader

The web reader fetches text/HTML pages and extracts readable text with Python's standard `html.parser`. It intentionally avoids executing JavaScript.

## Repository tools

Repository mode uses local read-only tools:

- file indexing with generated-directory ignores;
- query-based file search;
- Python/JavaScript/TypeScript dependency extraction;
- static risk pattern scanning;
- architecture summary generation.

## Trace tools

The trace collector stores JSONL events:

- question;
- plan;
- tool calls;
- search results;
- sources;
- notes;
- verification;
- final answer.

These traces are the input to `training\prepare_sft_dataset.py`.
