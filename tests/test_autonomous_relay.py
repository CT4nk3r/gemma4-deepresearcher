import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from training.autonomous_relay import (
    adapter_ready,
    build_feedback,
    jsonl_count,
    latest_adapter_for_resume,
    paused,
    prune_checkpoints,
    run_attempt_name,
    train,
    write_adaptive_seed_examples,
)


class Args:
    pause_file = ""


class AutonomousRelayTests(unittest.TestCase):
    def test_jsonl_count_missing_file_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(jsonl_count(Path(tmp) / "missing.jsonl"), 0)

    def test_paused_uses_pause_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pause_file = Path(tmp) / "pause"
            args = Args()
            args.pause_file = str(pause_file)
            self.assertFalse(paused(args))
            pause_file.write_text("pause", encoding="utf-8")
            self.assertTrue(paused(args))

    def test_run_attempt_name_adds_suffix_when_retries_enabled(self):
        self.assertEqual(run_attempt_name("run", 2, 3), "run-try02")
        self.assertEqual(run_attempt_name("run", 1, 1), "run")

    def test_prune_checkpoints_keeps_final_adapter_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint = run_dir / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "optimizer.pt").write_text("state", encoding="utf-8")
            adapter = run_dir / "adapter_model.safetensors"
            adapter.write_text("adapter", encoding="utf-8")

            prune_checkpoints(run_dir)

            self.assertFalse(checkpoint.exists())
            self.assertTrue(adapter.exists())

    def test_latest_adapter_for_resume_requires_peft_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest"
            args = SimpleNamespace(cumulative_adapter=True, latest_adapter_dir=str(latest))

            self.assertFalse(adapter_ready(latest))
            self.assertIsNone(latest_adapter_for_resume(args))

            latest.mkdir()
            (latest / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(adapter_ready(latest))
            self.assertIsNone(latest_adapter_for_resume(args))

            (latest / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
            self.assertTrue(adapter_ready(latest))
            self.assertEqual(latest_adapter_for_resume(args), str(latest))

    def test_latest_adapter_for_resume_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest"
            latest.mkdir()
            (latest / "adapter_config.json").write_text("{}", encoding="utf-8")
            (latest / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
            args = SimpleNamespace(cumulative_adapter=False, latest_adapter_dir=str(latest))

            self.assertIsNone(latest_adapter_for_resume(args))

    def test_train_retries_transient_training_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                base_model="google/gemma-4-e4b-it",
                clean_output=str(Path(tmp) / "clean.jsonl"),
                cumulative_adapter=True,
                latest_adapter_dir=str(Path(tmp) / "latest"),
                learning_rate=5e-5,
                log_output=str(Path(tmp) / "relay.log"),
                lora_alpha=32,
                lora_dropout=0.05,
                lora_r=16,
                pause_file=str(Path(tmp) / "pause"),
                raw_output=str(Path(tmp) / "raw.jsonl"),
                runs_dir=str(Path(tmp) / "runs"),
                status_output=str(Path(tmp) / "status.json"),
                teacher_model="teacher",
                teacher_target=1000,
                train_batch_size=1,
                train_grad_accum=2,
                train_max_length=512,
                train_retries=1,
                train_retry_sleep=0,
                train_steps=60,
                venv_python="python",
            )
            Path(args.clean_output).write_text("{}\n", encoding="utf-8")
            latest = Path(args.latest_adapter_dir)
            latest.mkdir()
            (latest / "adapter_config.json").write_text("{}", encoding="utf-8")
            (latest / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
            calls = []

            def fake_train_once(_args, _deadline, output_dir, *, resume_adapter=None):
                calls.append((output_dir, resume_adapter))
                if len(calls) == 1:
                    raise RuntimeError("transient native crash")

            with (
                patch("training.autonomous_relay.unload_teacher"),
                patch("training.autonomous_relay.train_once", side_effect=fake_train_once),
                patch("training.autonomous_relay.shutil.rmtree"),
                patch("training.autonomous_relay.shutil.copytree"),
            ):
                train(args, datetime.now(timezone.utc) + timedelta(minutes=5), cycle=6)

            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0][0].endswith("-try01"))
            self.assertTrue(calls[1][0].endswith("-try02"))
            self.assertEqual(calls[0][1], str(latest))
            self.assertEqual(calls[1][1], str(latest))

    def test_build_feedback_prioritizes_hallucination_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "delta": {
                                "overall_score": -0.02,
                                "citation_rate": 0.01,
                                "hallucination_proxy": 0.29,
                                "uncertainty_score": 0.2,
                                "format_score": 0.07,
                            }
                        },
                        "items": [
                            {
                                "index": 1,
                                "question": "What evidence is needed?",
                                "delta": {"overall_score": -0.1, "hallucination_proxy": 0.5},
                                "adapter": {"response": "Direct answer: unsupported detail."},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            feedback = build_feedback(eval_path, seed_count=1)

            self.assertIn("Reduce hallucination_proxy", feedback["focus_text"])
            self.assertEqual(len(feedback["worst_items"]), 1)

    def test_write_adaptive_seed_examples_uses_eval_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "seed.jsonl"
            feedback = {
                "priorities": ["Reduce hallucination_proxy"],
                "worst_items": [
                    {
                        "index": 7,
                        "question": "Assess a claim.",
                        "adapter": {"response": "Direct answer: The claim is true."},
                    }
                ],
            }

            write_adaptive_seed_examples(feedback, output)

            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["metadata"]["repair_target"], "hallucination_proxy")
            self.assertIn("[S1]", row["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
