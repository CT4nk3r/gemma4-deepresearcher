# Reproduction Plan

## Public DeepResearch methodology

The public Alibaba-NLP/DeepResearch codebase exposes a ReAct-style inference loop, search and visit tools, WebWeaver planner/writer prompts, WebResearcher evolving report memory, ReSum summarization memory, AgentFold compression, and evaluation scripts. The fully trained model behavior and some heavy-mode training details are not fully reproducible from public code alone.

## Local reproduction scope

This project reproduces the framework-level behavior that can be implemented locally:

- OpenAI-compatible or Ollama model calls;
- planner/search/read/note/verify/report stages;
- JSONL trace collection;
- citation/source validation;
- repository analysis mode;
- future LoRA training from curated traces.

It does not claim to reproduce Tongyi DeepResearch model weights, training data, hidden chain-of-thought behavior, or long-horizon RL behavior.

## Steps

1. Install the project in editable mode.
2. Start LM Studio or Ollama with a Gemma E4B-class instruction model.
3. Run a web research task with tracing enabled.
4. Inspect citations and source quality.
5. Run repository-mode tasks.
6. Convert successful traces into SFT JSONL.
7. Curate examples and train a LoRA adapter.
8. Re-run benchmarks and compare citation and completion metrics.

## Adaptation notes

Gemma E4B should use explicit external state rather than model-internal long-context planning. If quality is poor, improve deterministic note extraction, reduce iteration breadth, curate traces, and fine-tune citation/report behavior before adding more autonomous tool calls.
