import tempfile
import unittest
from pathlib import Path

from training.publish_adapter import validate_adapter_dir


class PublishAdapterTests(unittest.TestCase):
    def test_validates_adapter_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
            validate_adapter_dir(path)

    def test_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                validate_adapter_dir(path)


if __name__ == "__main__":
    unittest.main()
