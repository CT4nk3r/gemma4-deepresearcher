---
license: other
base_model: google/gemma-4-e4b-it
library_name: peft
tags:
  - gemma
  - lora
  - research-agent
  - deep-research
  - citation
---

# Gemma4 E4B Researcher LoRA

This is an experimental LoRA adapter that specializes `google/gemma-4-e4b-it` for research-assistant behavior: evidence-aware answers, citation discipline, uncertainty when evidence is insufficient, and concise synthesis.

This adapter is not an official Google, Alibaba, Tongyi, or DeepResearch release.

## Training summary

- **Base model:** `google/gemma-4-e4b-it`
- **Method:** supervised fine-tuning with PEFT/LoRA
- **Teacher model:** Alibaba-NLP Tongyi DeepResearch served locally through LM Studio
- **Bootstrap data:** public WebGPT-style cited QA examples converted into chat SFT format
- **Distillation goal:** rewrite examples into source-grounded researcher responses with explicit citations and insufficient-evidence behavior

## Intended use

Use this adapter with the Gemma4 E4B base model for local research-assistant experiments where the application supplies sources, notes, or retrieved evidence and expects grounded synthesis.

## Limitations

- The adapter does not add factual knowledge by itself.
- The model can still hallucinate, misuse citations, or overstate weak evidence.
- It should be paired with retrieval, source tracking, and citation validation.
- It is not suitable for high-stakes research without human review.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "google/gemma-4-e4b-it"
adapter = "<your-username>/<repo-name>"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)
```

## License and data notes

Users must comply with the base model license and the terms of any datasets used during training. The training data may include public cited-QA examples and teacher-distilled outputs; verify redistribution rights before publishing data.
