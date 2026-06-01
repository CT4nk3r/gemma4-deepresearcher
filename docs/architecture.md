# Architecture

This project adapts the public DeepResearch pattern for a smaller local Gemma E4B-class model. The core design choice is to move deterministic orchestration into Python and keep the model focused on compact planning and synthesis.

## Components

| Component | Implementation | Responsibility |
|---|---|---|
| CLI | `src\gemma_research\cli.py` | Parse questions, config, repository mode, trace/output options. |
| Config | `src\gemma_research\config.py` | Load TOML and runtime overrides for model, search, agent, and repository settings. |
| Model client | `src\gemma_research\llm.py` | Support LM Studio/OpenAI-compatible chat, Ollama native chat, and deterministic offline mode. |
| Planner | `ResearchAgent._create_plan` | Generate a compact plan and search queries. Offline mode uses deterministic keywords. |
| Searcher | `src\gemma_research\search.py` | Search DuckDuckGo HTML or disable search for offline tests. |
| Reader | `src\gemma_research\readers.py` | Fetch and normalize web pages into text documents. |
| Note extractor | `src\gemma_research\notes.py` | Extract keyword-relevant source-backed notes. |
| Verifier | `src\gemma_research\verification.py` | Check source count, evidence coverage, and citation validity. |
| Report writer | `src\gemma_research\report.py` | Ask the model for cited markdown or use deterministic fallback. |
| Trace collector | `src\gemma_research\tracing.py` | Store JSONL events for later SFT dataset preparation. |
| Repository analyzer | `src\gemma_research\repository.py` | Read-only indexing, search, dependency signals, risk signals, architecture summary. |

## DeepResearch influence

Alibaba-NLP/DeepResearch uses a ReAct loop with XML-tagged tool calls, search/visit tools, append-only conversation memory, and separate WebAgent variants that introduce evolving reports, outline planning, and external memory banks. The public repository includes OpenAI-compatible inference, search/visit tooling, WebWeaver planner/writer prompts, WebResearcher evolving report memory, and ReSum/AgentFold compression approaches.

This implementation borrows the framework-level ideas rather than relying on model-specific behavior:

- multi-step research loop;
- external source memory;
- source IDs and citation validation;
- explicit gap detection;
- trace collection;
- replaceable search, read, model, and repository tools.

It intentionally does not depend on Alibaba-specific trained tags such as `<think>`, `<answer>`, `<qwen:cite>`, or long-horizon 100-turn behavior because those are model-dependent and not reliable for a local Gemma E4B-class model without additional fine-tuning.

## Local model strategy

Gemma E4B should be treated as a bounded reasoning and synthesis component. Python owns iteration limits, source tracking, JSON repair, repository parsing, and metric scoring. This makes the system usable through LM Studio or Ollama even when the model is not trained for the exact DeepResearch XML protocol.
