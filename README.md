# Gemma DeepResearch E4B

Local DeepResearch-style research agent for Gemma E4B-class models. The runtime keeps orchestration deterministic in Python and uses a local model through LM Studio, Ollama, or any OpenAI-compatible endpoint.

## Quick start

```powershell
python -m pip install -e .
gemma-research --config configs\lmstudio.toml "What are the tradeoffs of local RAG?"
gemma-research --repo . "Analyze this codebase"
```

LM Studio defaults to `http://localhost:1234/v1`. Ollama defaults to `http://localhost:11434` and uses its native chat API.

## Highlights

- Structured planning, query generation, source reading, note extraction, verification, and cited report writing.
- Read-only repository research mode with indexing, search, dependency extraction, risk scanning, and architecture summaries.
- Automatic JSONL trace collection for future supervised fine-tuning and LoRA workflows.
- Guardrails for citation validation, source tracking, maximum iteration limits, insufficient-evidence detection, and JSON repair.
- Public-data and LM Studio teacher-distillation utilities for researcher-specialist SFT bootstrapping.
- Hugging Face adapter publishing helpers and model-card template.

## Local smoke test

The offline provider is deterministic and useful for development:

```powershell
gemma-research --provider offline --search-provider none "Explain this project"
gemma-research --provider offline --repo . "Analyze this repository"
```
