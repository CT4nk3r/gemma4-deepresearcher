from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a trained LoRA adapter to Hugging Face.")
    parser.add_argument("--adapter-dir", required=True, help="Directory produced by train_lora.py")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/gemma4-e4b-researcher-lora")
    parser.add_argument("--model-card", default="publishing\\hf_adapter_model_card.md")
    parser.add_argument("--private", action="store_true", help="Create or update a private repo")
    parser.add_argument("--dry-run", action="store_true", help="Validate files without uploading")
    parser.add_argument("--commit-message", default="Publish Gemma4 E4B researcher LoRA adapter")
    args = parser.parse_args(argv)

    adapter_dir = Path(args.adapter_dir)
    validate_adapter_dir(adapter_dir)
    model_card = Path(args.model_card)
    if not model_card.exists():
        raise SystemExit(f"Model card template not found: {model_card}")

    readme_path = adapter_dir / "README.md"
    if not readme_path.exists():
        shutil.copyfile(model_card, readme_path)

    if args.dry_run:
        print(f"Validated adapter directory: {adapter_dir}")
        print(f"Would publish to: {args.repo_id}")
        print(f"Model card: {readme_path}")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit('Publishing requires: python -m pip install -e ".[training]"') from exc

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        commit_message=args.commit_message,
    )
    print(f"Published adapter to https://huggingface.co/{args.repo_id}")
    return 0


def validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter_dir}")
    required = ["adapter_config.json"]
    missing = [name for name in required if not (adapter_dir / name).exists()]
    has_weights = any(
        (adapter_dir / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if missing:
        raise SystemExit(f"Adapter directory missing required file(s): {', '.join(missing)}")
    if not has_weights:
        raise SystemExit("Adapter directory missing adapter_model.safetensors or adapter_model.bin")


if __name__ == "__main__":
    raise SystemExit(main())
