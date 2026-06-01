import unittest

from training.prepare_uncertainty_repair_sft import build_examples


class PrepareUncertaintyRepairSftTests(unittest.TestCase):
    def test_build_examples_are_chat_sft_shape(self):
        examples = list(build_examples(repeat=1))

        self.assertGreaterEqual(len(examples), 20)
        first = examples[0]
        self.assertEqual([message["role"] for message in first["messages"]], ["system", "user", "assistant"])
        self.assertIn("uncertainty-repair-seed", first["metadata"]["dataset_id"])


if __name__ == "__main__":
    unittest.main()
