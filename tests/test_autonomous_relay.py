import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from training.autonomous_relay import jsonl_count, paused, run_attempt_name, train


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

    def test_train_retries_transient_training_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                base_model="google/gemma-4-e4b-it",
                clean_output=str(Path(tmp) / "clean.jsonl"),
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
            calls = []

            def fake_train_once(_args, _deadline, output_dir):
                calls.append(output_dir)
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
            self.assertTrue(calls[0].endswith("-try01"))
            self.assertTrue(calls[1].endswith("-try02"))


if __name__ == "__main__":
    unittest.main()
