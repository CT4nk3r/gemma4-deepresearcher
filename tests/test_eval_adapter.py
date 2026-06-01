import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import eval_adapter


class EvalAdapterTests(unittest.TestCase):
    def test_read_eval_set_validates_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question": "What evidence is needed?",
                        "expected_behaviors": ["states uncertainty"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = eval_adapter.read_eval_set(path)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].question, "What evidence is needed?")
            self.assertEqual(examples[0].expected_behaviors, ["states uncertainty"])

    def test_score_response_tracks_citations_uncertainty_and_format(self):
        response = (
            "Direct answer: The trial reduced symptoms by 20 percent [S1].\n\n"
            "Evidence summary: The study reported fewer symptoms in the intervention group [S1]. "
            "The company press release is more promotional.\n\n"
            "Conclusion: The result is promising, but the evidence is insufficient for a firm causal claim."
        )

        metrics = eval_adapter.score_response(response, ["states uncertainty", "cites sources"])

        self.assertGreater(metrics["citation_rate"], 0)
        self.assertGreater(metrics["hallucination_proxy"], 0)
        self.assertTrue(metrics["uncertainty_present"])
        self.assertEqual(metrics["format_score"], 1.0)

    def test_main_dry_run_writes_outputs_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_set = tmp_path / "eval.jsonl"
            output_dir = tmp_path / "results"
            eval_set.write_text(
                json.dumps(
                    {
                        "question": "How should uncertainty be handled?",
                        "expected_behaviors": ["states uncertainty", "format compliance"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(eval_adapter, "load_model_bundle") as load_model:
                result = eval_adapter.main(
                    [
                        "--adapter",
                        str(tmp_path / "missing-adapter"),
                        "--base-model",
                        "google/gemma-4-e4b-it",
                        "--eval-set",
                        str(eval_set),
                        "--output-dir",
                        str(output_dir),
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            load_model.assert_not_called()
            json_results = list(output_dir.glob("eval-*.json"))
            markdown_results = list(output_dir.glob("eval-*.md"))
            self.assertEqual(len(json_results), 1)
            self.assertEqual(len(markdown_results), 1)
            payload = json.loads(json_results[0].read_text(encoding="utf-8"))
            self.assertIn("aggregate", payload)
            self.assertEqual(len(payload["items"]), 1)

    def test_validate_adapter_dir_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit):
                eval_adapter.validate_adapter_dir(path)


if __name__ == "__main__":
    unittest.main()
