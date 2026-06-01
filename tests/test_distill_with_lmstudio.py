import unittest

from training.distill_with_lmstudio import (
    _skip,
    _take,
    build_distillation_prompt,
    distill_example,
    is_retryable_error,
    keep_original,
)


class DistillWithLMStudioTests(unittest.TestCase):
    def test_build_prompt_contains_source_rules(self):
        prompt = build_distillation_prompt('{"question":"Q","sources":[]}', "Answer")
        self.assertIn("Cite every factual claim", prompt)
        self.assertIn("Compare sources", prompt)
        self.assertIn("Do not invent", prompt)

    def test_dry_run_distills_example_shape(self):
        example = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": '{"question":"Q"}'},
                {"role": "assistant", "content": "Answer [1]"},
            ],
            "metadata": {"dataset_id": "test"},
        }
        distilled = distill_example(
            example,
            base_url="http://localhost:1234/v1",
            model="teacher",
            temperature=0.2,
            max_tokens=128,
            timeout=30,
            dry_run=True,
        )

        self.assertEqual(distilled["messages"][-1]["role"], "assistant")
        self.assertIn("distilled_by", distilled["metadata"])

    def test_keep_original_marks_failure(self):
        example = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        kept = keep_original(example, model="teacher", dry_run=False)
        self.assertEqual(kept["messages"][-1]["content"], "answer")
        self.assertEqual(kept["metadata"]["distilled_by"], "original")

    def test_skip_and_take_for_resume(self):
        values = iter([{"id": 1}, {"id": 2}, {"id": 3}])
        self.assertEqual(list(_take(_skip(values, 1), 1)), [{"id": 2}])

    def test_retryable_error_detection(self):
        self.assertTrue(is_retryable_error("LM Studio HTTP 400: Model unloaded."))
        self.assertFalse(is_retryable_error("invalid API key"))


if __name__ == "__main__":
    unittest.main()
