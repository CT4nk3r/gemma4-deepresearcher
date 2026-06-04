# Training

The public DeepResearch repository does not release the full training data synthesis pipeline. For this project, fine-tuning starts with public bootstrap datasets and then improves with traces produced by successful local runs.

## Public bootstrap datasets

The public-data manifest is `configs\public_datasets.json`. It currently includes WebGPT comparisons, HotpotQA, MuSiQue, and generic agent/tool-use conversations. These do not reproduce Alibaba's private data, but they are useful for bootstrapping citation discipline, evidence-backed QA, multi-hop reasoning, and structured agent behavior.

Install the optional data dependency:

```powershell
python -m pip install -e ".[data]"
```

Build an initial researcher SFT file:

```powershell
python training\prepare_public_sft.py --dataset webgpt-comparisons --max-examples 1000 --output data\public_bootstrap_sft.jsonl
```

`webgpt-comparisons` streams from OpenAI's public JSONL URL and works without the Hugging Face dataset loader. Some older Hugging Face datasets, including HotpotQA, use dataset loading scripts. The manifest marks those with `trust_remote_code: true`; only run them when you trust the dataset repository. If scripted Hugging Face datasets fail on Python 3.14, use Python 3.11 or 3.12 for data preparation.

For a local JSONL fixture or manually curated rows:

```powershell
python training\prepare_public_sft.py --local-jsonl data\curated.jsonl --converter instruction_response --output data\curated_sft.jsonl
```

Review dataset cards, licenses, and redistribution rules before using or sharing converted data.

## Data collection

Run the agent with tracing enabled:

```powershell
gemma-research --config configs\lmstudio.toml "Research question"
```

Traces are written as JSONL under `agent.trace_dir`, default `.gemma-research\traces`.

## SFT dataset preparation

```powershell
python training\prepare_sft_dataset.py .gemma-research\traces --output data\sft.jsonl
```

The converter builds chat examples containing the question, plan, notes, and final answer. Curate the output before training; remove failed traces, weak citations, and low-quality reports.

The best first training mix is:

1. Public bootstrap SFT for broad citation/evidence/tool behavior.
2. Curated traces from this exact runtime for matching prompts and tool schemas.
3. Rejection-sampled traces from stronger models for hard research tasks.

## LM Studio teacher distillation

LM Studio can run teacher inference with ROCm, but it does not train LoRA adapters. Use it to improve the SFT data before training:

```powershell
python training\distill_with_lmstudio.py --input data\public_bootstrap_sft.jsonl --output data\teacher_distilled_sft.jsonl --model alibaba-nlp_tongyi-deepresearch-30b-a3b --max-examples 100
```

See `docs\amd-training.md` for the AMD-specific workflow.

Validate the distilled data before training:

```powershell
python training\validate_sft_dataset.py data\teacher_distilled_sft.jsonl --strict
```

If validation finds a few weak rows, create a clean training file:

```powershell
python training\clean_sft_dataset.py --input data\teacher_distilled_sft.jsonl --output data\teacher_distilled_clean_sft.jsonl
```

The cleaner drops invalid citation rows and duplicate user prompts by default.

## Autonomous distillation scripts

Use the Windows wrappers when you want the teacher-data job to run unattended:

```powershell
scripts\start-teacher-distillation.ps1 -Target 500
scripts\status-teacher-distillation.ps1
scripts\stop-teacher-distillation.ps1
```

The supervisor writes logs and status under `.gemma-research\`. It resumes from the existing raw dataset, cleans and validates after each chunk, and skips persistent empty teacher completions.

If PowerShell blocks local scripts, run the same commands through `powershell -ExecutionPolicy Bypass -File ...`. The supervisor also stops instead of spinning forever if several chunks produce no new examples.

## Autonomous teach/train relay

On the local RX 7600 XT machine, do not keep the Tongyi teacher and Gemma4 student loaded at the same time. The relay alternates phases: Tongyi generates more cited examples, the cleaner/validator filters them, Tongyi unloads, then Gemma4 trains a LoRA pass.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-autonomous-relay.ps1 -Hours 18
powershell -ExecutionPolicy Bypass -File scripts\status-autonomous-relay.ps1
powershell -ExecutionPolicy Bypass -File scripts\stop-autonomous-relay.ps1
```

The latest successful adapter is copied to `runs\gemma4-e4b-deepresearch-lora-latest`. Individual pass outputs remain under `runs\gemma4-e4b-deepresearch-lora-relay-*`.

By default, `scripts\start-autonomous-relay.ps1` now starts the relay in closed-loop mode:

1. Tongyi distills new learning material.
2. Gemma4 continues training from `runs\gemma4-e4b-deepresearch-lora-latest`.
3. The relay evaluates base Gemma4 vs. the latest adapter.
4. Eval regressions are written to `.gemma-research\closed_loop_feedback.json` and `.gemma-research\closed_loop_focus.txt`.
5. The next teacher pass uses that feedback to create targeted repair examples, especially for `hallucination_proxy`.

Use `-DisableClosedLoop` when you need a training-only relay.

## LoRA training

```powershell
python training\train_lora.py --model google/gemma-4-e4b-it --dataset data\teacher_distilled_clean_sft.jsonl --output-dir adapters\gemma-research-lora --fp16 --gradient-checkpointing --max-length 2048 --batch-size 1 --gradient-accumulation-steps 16 --max-steps 200
```

The script expects optional packages from `python -m pip install -e ".[training]"`. Use trainable Hugging Face/safetensors weights, not LM Studio GGUF files. GGUF files are inference quantizations and cannot be trained by the Transformers LoRA script. On AMD, avoid `--load-in-4bit` unless your ROCm bitsandbytes build explicitly supports it.

## Merge adapter

```powershell
python training\merge_adapter.py --model <base-gemma-model> --adapter adapters\gemma-research-lora --output-dir models\gemma-research-merged
```

## Target behaviors

- compact planning;
- tool-use discipline;
- citation discipline;
- gap identification;
- refusal to over-answer when evidence is weak;
- synthesis from source-backed notes.

## Next training priority

The next relay pass should explicitly target `hallucination_proxy` as the main regression to fix.
That means:

- favoring cited claims over uncited elaboration;
- penalizing answers that add factual details without support;
- keeping the uncertainty behavior we gained, but making sure it does not come with extra unsupported claims;
- reviewing eval failures where the adapter was more verbose or more confident than the base model without better evidence.
