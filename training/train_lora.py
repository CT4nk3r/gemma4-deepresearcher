from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
GEMMA4_LANGUAGE_TARGET_REGEX = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter from SFT JSONL.")
    parser.add_argument("--model", required=True, help="Base Gemma model path or Hugging Face id")
    parser.add_argument("--dataset", required=True, help="SFT JSONL produced by prepare_sft_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Adapter output directory")
    parser.add_argument("--resume-adapter", help="Existing PEFT LoRA adapter directory to continue training")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help="Comma-separated module names to adapt, or regex:<pattern>",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA 4-bit loading")
    parser.add_argument("--bf16", action="store_true", help="Train with bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Train with float16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval; use 0 to save only the final adapter")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--train-on-input", action="store_true", help="Do not mask prompt tokens")
    args = parser.parse_args(argv)

    _reject_gguf_model(args.model)
    _install_transformers_continuous_batching_shim()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
        import torch
    except ImportError as exc:
        raise SystemExit(
            'LoRA training requires optional packages: python -m pip install -e ".[training]"'
        ) from exc

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if args.resume_adapter and not adapter_ready(args.resume_adapter):
        raise SystemExit(f"Resume adapter is missing required PEFT files: {args.resume_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("4-bit QLoRA requires bitsandbytes and a supported GPU environment.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model_config = AutoConfig.from_pretrained(args.model)
    model_loader = _model_loader_for_config(model_config, AutoModelForCausalLM, AutoModelForImageTextToText)
    model = model_loader.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=_target_modules(args.target_modules, model_config),
                task_type="CAUSAL_LM",
            ),
        )
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        if args.train_on_input:
            tokens["labels"] = tokens["input_ids"].copy()
        else:
            assistant_text = str(example["messages"][-1]["content"])
            tokens["labels"] = _assistant_only_labels(tokenizer, tokens["input_ids"], assistant_text)
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="no" if args.save_steps <= 0 else "steps",
        save_total_limit=1,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


def _reject_gguf_model(model: str) -> None:
    path = Path(model)
    if path.suffix.lower() == ".gguf":
        raise SystemExit(
            "GGUF files are inference quantizations and cannot be trained with this "
            "Transformers LoRA script. Use the trainable Hugging Face id "
            "google/gemma-4-e4b-it or a local safetensors model directory."
        )
    if path.exists() and path.is_dir():
        ggufs = list(path.glob("*.gguf"))
        safetensors = list(path.glob("*.safetensors"))
        if ggufs and not safetensors:
            raise SystemExit(
                "This model directory contains GGUF files only. Use google/gemma-4-e4b-it "
                "or download the trainable safetensors weights."
            )


def adapter_ready(path: str | Path) -> bool:
    adapter = Path(path)
    return (
        adapter.is_dir()
        and (adapter / "adapter_config.json").is_file()
        and (adapter / "adapter_model.safetensors").is_file()
    )


def _target_modules(value: str, model_config: Any | None = None) -> list[str] | str:
    if value.startswith("regex:"):
        regex = value.removeprefix("regex:").strip()
        if not regex:
            raise SystemExit("--target-modules regex must not be empty")
        return regex
    if _is_gemma4_config(model_config) and value == DEFAULT_TARGET_MODULES:
        return GEMMA4_LANGUAGE_TARGET_REGEX
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise SystemExit("--target-modules must contain at least one module name")
    return modules


def _install_transformers_continuous_batching_shim() -> None:
    """Disable optional Transformers continuous batching on ROCm Windows builds.

    AMD's Windows ROCm PyTorch wheels currently omit distributed C10d support.
    Transformers 5 imports its optional continuous batching path while loading
    Gemma4, and that path imports DTensor/distributed modules even for normal
    single-GPU training. Training does not use continuous batching, so a small
    no-op module keeps the Gemma4 model importable without patching site-packages.
    """

    module_name = "transformers.generation.continuous_batching"
    if module_name in sys.modules:
        return

    class ContinuousMixin:
        pass

    class UnsupportedContinuousBatching:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Transformers continuous batching is disabled for this ROCm Windows "
                "training environment."
            )

    def add_module(name: str, **attrs: Any) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        sys.modules[name] = module
        return module

    base = add_module(module_name, ContinuousMixin=ContinuousMixin)
    for attr_name in [
        "ContinuousBatchingManager",
        "FIFOScheduler",
        "PagedAttentionCache",
        "PrefillFirstScheduler",
        "RequestState",
        "RequestStatus",
        "Scheduler",
    ]:
        setattr(base, attr_name, UnsupportedContinuousBatching)

    add_module(
        f"{module_name}.cache",
        PagedAttentionCache=UnsupportedContinuousBatching,
    )
    add_module(
        f"{module_name}.continuous_api",
        ContinuousBatchingManager=UnsupportedContinuousBatching,
        ContinuousMixin=ContinuousMixin,
    )
    add_module(
        f"{module_name}.requests",
        RequestState=UnsupportedContinuousBatching,
        RequestStatus=UnsupportedContinuousBatching,
    )
    add_module(
        f"{module_name}.scheduler",
        FIFOScheduler=UnsupportedContinuousBatching,
        PrefillFirstScheduler=UnsupportedContinuousBatching,
        Scheduler=UnsupportedContinuousBatching,
    )


def _model_loader(
    model: str,
    auto_config: Any,
    causal_lm_loader: Any,
    image_text_loader: Any,
) -> Any:
    config = auto_config.from_pretrained(model)
    return _model_loader_for_config(config, causal_lm_loader, image_text_loader)


def _model_loader_for_config(
    config: Any,
    causal_lm_loader: Any,
    image_text_loader: Any,
) -> Any:
    if _is_gemma4_config(config):
        return image_text_loader
    return causal_lm_loader


def _is_gemma4_config(config: Any | None) -> bool:
    if config is None:
        return False
    architectures = set(getattr(config, "architectures", []) or [])
    return getattr(config, "model_type", None) == "gemma4" or "Gemma4ForConditionalGeneration" in architectures


def _assistant_only_labels(tokenizer, input_ids: list[int], assistant_text: str) -> list[int]:
    labels = [-100] * len(input_ids)
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    start = _find_last_subsequence(input_ids, assistant_ids)
    if start == -1:
        return input_ids.copy()
    for index in range(start, min(start + len(assistant_ids), len(input_ids))):
        if input_ids[index] != tokenizer.pad_token_id:
            labels[index] = input_ids[index]
    return labels


def _find_last_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
