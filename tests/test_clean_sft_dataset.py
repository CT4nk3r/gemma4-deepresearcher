import json
import tempfile
import unittest
from pathlib import Path

from training.clean_sft_dataset import main


class CleanSFTDatasetTests(unittest.TestCase):
    def test_drops_invalid_rows(self):
        good = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"sources": [{"id": "1", "title": "T", "text": "Evidence"}]}
                    ),
                },
                {"role": "assistant", "content": "Answer [1]"},
            ]
        }
        bad = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"sources": [{"id": "1", "title": "T", "text": "Evidence"}]}
                    ),
                },
                {"role": "assistant", "content": "Answer without citation"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                json.dumps(good) + "\n" + json.dumps(bad) + "\n",
                encoding="utf-8",
            )
            code = main(["--input", str(input_path), "--output", str(output_path)])
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1)

    def test_drops_duplicate_user_prompts(self):
        example = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"sources": [{"id": "1", "title": "T", "text": "Evidence"}]}
                    ),
                },
                {"role": "assistant", "content": "Answer [1]"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                json.dumps(example) + "\n" + json.dumps(example) + "\n",
                encoding="utf-8",
            )
            code = main(["--input", str(input_path), "--output", str(output_path)])
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
