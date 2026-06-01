import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import supervise_distillation


class SuperviseDistillationTests(unittest.TestCase):
    def test_missing_raw_count_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(supervise_distillation.raw_count(Path(tmp) / "missing.jsonl"), 0)

    def test_stops_after_stalled_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            status_path = tmp_path / "status.json"
            with (
                mock.patch.object(supervise_distillation, "distill_main", return_value=0),
                mock.patch.object(supervise_distillation, "clean_and_validate", return_value=None),
                self.assertRaises(RuntimeError),
            ):
                supervise_distillation.main(
                    [
                        "--input",
                        str(tmp_path / "input.jsonl"),
                        "--raw-output",
                        str(tmp_path / "raw.jsonl"),
                        "--clean-output",
                        str(tmp_path / "clean.jsonl"),
                        "--status-output",
                        str(status_path),
                        "--pause-file",
                        str(tmp_path / "pause"),
                        "--target",
                        "1",
                        "--chunk-size",
                        "1",
                        "--failure-sleep",
                        "0",
                        "--max-stalled-chunks",
                        "1",
                    ]
                )

            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "stalled")
            self.assertEqual(status["stalled_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
