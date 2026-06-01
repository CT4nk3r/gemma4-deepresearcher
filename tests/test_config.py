import tempfile
import unittest
from pathlib import Path

from gemma_research.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_toml_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[model]\nprovider = "offline"\n', encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.model.provider, "offline")

    def test_rejects_unknown_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[unknown]\nvalue = 1\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
