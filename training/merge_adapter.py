from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--model", required=True, help="Base Gemma model path or Hugging Face id")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory")
    args = parser.parse_args(argv)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Merging requires optional packages: transformers, peft, accelerate.") from exc

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
