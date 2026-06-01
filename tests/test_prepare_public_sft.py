import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_public_sft import convert_row, iter_jsonl, validate_sft_example, write_jsonl


class PreparePublicSFTTests(unittest.TestCase):
    def test_converts_webgpt_comparison_by_higher_score(self):
        example = convert_row(
            {
                "question": {"full_text": "What is retrieval augmented generation?"},
                "answer_0": "Weak answer.",
                "answer_1": "Grounded answer with citation [1].",
                "quotes": [{"title": "Source title", "extract": "Useful evidence."}],
                "score_0": 0.1,
                "score_1": 0.9,
            },
            converter="webgpt_comparisons",
            metadata={"dataset_id": "webgpt"},
        )

        self.assertIsNotNone(example)
        validate_sft_example(example)
        self.assertIn("Grounded answer", example["messages"][-1]["content"])
        self.assertIn('"id": "1"', example["messages"][1]["content"])

    def test_converts_hotpot_style_evidence_qa(self):
        example = convert_row(
            {
                "question": "Where was the creator of Python born?",
                "answer": "The Netherlands",
                "context": {
                    "title": ["Python", "Guido van Rossum"],
                    "sentences": [
                        ["Python is a programming language."],
                        ["Guido van Rossum was born in the Netherlands."],
                    ],
                },
                "supporting_facts": {"title": ["Guido van Rossum"], "sent_id": [0]},
            },
            converter="evidence_qa",
            metadata={"dataset_id": "hotpotqa"},
        )

        self.assertIsNotNone(example)
        validate_sft_example(example)
        self.assertIn("[S2]", example["messages"][-1]["content"])

    def test_writes_jsonl_from_local_rows(self):
        rows = [
            [
                {
                    "messages": [
                        {"role": "user", "content": "Search for evidence."},
                        {"role": "assistant", "content": "I found evidence [S1]."},
                    ]
                }
            ]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            examples = (
                convert_row(row, converter="conversation", metadata={"dataset_id": "local"})
                for row in iter_jsonl(input_path)
            )
            count = write_jsonl((example for example in examples if example), output_path)

            self.assertEqual(count, 1)
            self.assertTrue(output_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
