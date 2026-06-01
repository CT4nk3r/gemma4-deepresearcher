# Publishing the Researcher Adapter

The recommended release artifact is a PEFT/LoRA adapter, not a full merged Gemma model. This keeps the upload small and avoids redistributing base weights.

## Publish flow

1. Train the adapter:

```powershell
python training\train_lora.py --model google/gemma-4-e4b-it --dataset data\teacher_distilled_clean_sft.jsonl --output-dir adapters\gemma-research-lora --fp16 --gradient-checkpointing --max-length 2048 --batch-size 1 --gradient-accumulation-steps 16 --max-steps 200
```

2. Edit `publishing\hf_adapter_model_card.md` with final eval results, dataset counts, and license notes.

3. Dry-run validation:

```powershell
python training\publish_adapter.py --adapter-dir adapters\gemma-research-lora --repo-id <user-or-org>\gemma4-e4b-researcher-lora --dry-run
```

4. Publish:

```powershell
$env:HF_TOKEN = "<your-token>"
python training\publish_adapter.py --adapter-dir adapters\gemma-research-lora --repo-id <user-or-org>\gemma4-e4b-researcher-lora --private
```

Start private, run evals from the published adapter, then make the repo public when the license review and model card are complete.

## What not to publish by default

- Do not upload the full merged Gemma weights unless the base model license explicitly permits it.
- Do not publish distilled training data until dataset and teacher-output redistribution rights are reviewed.
- Do not claim affiliation with Google, Alibaba, Tongyi, or DeepResearch.
