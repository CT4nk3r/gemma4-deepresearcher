# Prompts

The public Alibaba-NLP/DeepResearch inference prompt defines a deep research assistant with JSON tool calls wrapped in XML tags. WebWeaver adds a separate planner and writer with citation-bearing outlines and Qwen-specific citation tags.

This project uses simpler prompts that are safer for Gemma E4B-class local models.

## Planning prompt

System:

```text
You are the planning component of a local research agent. Return strict JSON only, with no markdown.
```

User:

```text
Create a compact research plan and search queries for this question.
Return JSON with shape:
{"steps":[{"id":"1","goal":"..."}],"queries":["..."]}.
Question: {question}
```

## Report prompt

System:

```text
You write concise research reports grounded only in supplied notes. Every factual claim must cite one of the provided source ids in square brackets. Do not cite unknown ids. If evidence is insufficient, say so.
```

User content includes the question, allowed source IDs, evidence status, and source-backed notes.

## Prompting constraints

- Ask for strict JSON only when structured output is required.
- Keep allowed citations explicit.
- Prefer normal markdown citations like `[S1]` over model-specific XML citation tags.
- Do not rely on hidden chain-of-thought tags. Use explicit plans, notes, and verification state instead.
