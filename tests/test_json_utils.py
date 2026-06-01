import unittest

from gemma_research.json_utils import repair_json


class JSONUtilsTests(unittest.TestCase):
    def test_repairs_fenced_json_with_trailing_comma(self):
        data = repair_json('```json\n{"items": [1, 2,],}\n```')
        self.assertEqual(data, {"items": [1, 2]})

    def test_extracts_json_from_surrounding_text(self):
        data = repair_json('Here is JSON: {"ok": true} done')
        self.assertEqual(data, {"ok": True})


if __name__ == "__main__":
    unittest.main()
