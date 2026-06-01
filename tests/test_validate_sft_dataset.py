import json
import tempfile
import unittest
from pathlib import Path

from training.validate_sft_dataset import validate_dataset


class ValidateSFTDatasetTests(unittest.TestCase):
    def test_accepts_valid_source_citation(self):
        stats = _validate_one(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": "Q",
                                "sources": [{"id": "S1", "title": "T", "text": "Evidence"}],
                            }
                        ),
                    },
                    {"role": "assistant", "content": "Answer [S1]"},
                ]
            }
        )
        self.assertEqual(stats.issue_count, 0)

    def test_flags_invalid_citation(self):
        stats = _validate_one(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": "Q",
                                "sources": [{"id": "S1", "title": "T", "text": "Evidence"}],
                            }
                        ),
                    },
                    {"role": "assistant", "content": "Answer [S2]"},
                ]
            }
        )
        self.assertEqual(stats.issues[0].code, "invalid_citation")


def _validate_one(example):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        path.write_text(json.dumps(example) + "\n", encoding="utf-8")
        return validate_dataset(path)


if __name__ == "__main__":
    unittest.main()
