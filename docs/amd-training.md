# AMD and LM Studio Training Path

LM Studio ROCm support accelerates local inference for GGUF models. It does not expose gradients or a fine-tuning API, so it cannot directly train a LoRA adapter.

Use LM Studio for:

- running the current Gemma model;
- running a stronger teacher model such as Tongyi DeepResearch;
- generating or distilling SFT examples through the OpenAI-compatible API.

Use ROCm PyTorch/Transformers for:

- loading trainable Hugging Face or safetensors weights;
- applying PEFT/LoRA;
- saving an adapter or merged model.

## Current machine

This machine exposes an AMD Radeon RX 7600 XT with about 16 GB of OpenCL-visible memory. The LM Studio API is reachable at `http://localhost:1234/v1`, and the available model IDs include `google/gemma-4-e4b` and `alibaba-nlp_tongyi-deepresearch-30b-a3b`.

No ROCm command-line tools or PyTorch training stack are currently visible in this shell. Native Windows AMD support in LM Studio is inference-oriented. For LoRA training, use a Linux or WSL environment with ROCm PyTorch if the GPU/driver combination is supported.

## Distill with LM Studio teacher

Generate a public bootstrap dataset first:

```powershell
python training\prepare_public_sft.py --dataset webgpt-comparisons --max-examples 1000 --output data\public_bootstrap_sft.jsonl
```

Use Tongyi DeepResearch through LM Studio as a teacher to rewrite examples into the target researcher style:

```powershell
python training\distill_with_lmstudio.py --input data\public_bootstrap_sft.jsonl --output data\teacher_distilled_sft.jsonl --model alibaba-nlp_tongyi-deepresearch-30b-a3b --max-examples 100
```

Use Gemma itself for lower-cost self-distillation experiments:

```powershell
python training\distill_with_lmstudio.py --input data\public_bootstrap_sft.jsonl --output data\gemma_distilled_sft.jsonl --model google/gemma-4-e4b --max-examples 100
```

## LoRA on AMD ROCm

On a ROCm-capable Linux/WSL environment, install the training stack with a ROCm PyTorch wheel appropriate for your ROCm version, then:

```powershell
python training\train_lora.py --model google/gemma-4-e4b-it --dataset data\teacher_distilled_clean_sft.jsonl --output-dir adapters\gemma-research-lora --fp16 --gradient-checkpointing --max-length 2048 --batch-size 1 --gradient-accumulation-steps 16 --max-steps 200
```

Avoid `--load-in-4bit` on AMD unless your installed bitsandbytes build explicitly supports ROCm in your environment. For a 16 GB card, start with `--max-length 1024` or `2048` and increase only if memory allows.

## Back to LM Studio

After training, merge the adapter into safetensors:

```powershell
python training\merge_adapter.py --model google/gemma-4-e4b-it --adapter adapters\gemma-research-lora --output-dir models\gemma-research-merged
```

Then convert the merged Hugging Face model to GGUF with llama.cpp and import the GGUF into LM Studio.
