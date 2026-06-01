import tempfile
import unittest
from pathlib import Path

from training.autonomous_relay import jsonl_count, paused


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


if __name__ == "__main__":
    unittest.main()
