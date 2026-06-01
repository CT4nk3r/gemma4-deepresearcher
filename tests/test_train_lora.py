import sys
import tempfile
import unittest
from pathlib import Path

from training.train_lora import (
    GEMMA4_LANGUAGE_TARGET_REGEX,
    _find_last_subsequence,
    _install_transformers_continuous_batching_shim,
    _model_loader,
    _reject_gguf_model,
    _target_modules,
)


class TrainLoraTests(unittest.TestCase):
    def test_finds_last_subsequence(self):
        self.assertEqual(_find_last_subsequence([1, 2, 3, 2, 3], [2, 3]), 3)

    def test_target_modules_parse(self):
        self.assertEqual(_target_modules("q_proj, v_proj"), ["q_proj", "v_proj"])

    def test_target_modules_accept_regex(self):
        self.assertEqual(_target_modules("regex:.*language_model.*q_proj$"), ".*language_model.*q_proj$")

    def test_target_modules_use_gemma4_language_regex_for_default(self):
        class Config:
            model_type = "gemma4"
            architectures = ["Gemma4ForConditionalGeneration"]

        value = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.assertEqual(_target_modules(value, Config()), GEMMA4_LANGUAGE_TARGET_REGEX)

    def test_rejects_gguf_only_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "model.gguf").write_text("not real", encoding="utf-8")
            with self.assertRaises(SystemExit):
                _reject_gguf_model(str(path))

    def test_installs_transformers_continuous_batching_shim(self):
        for name in list(sys.modules):
            if name.startswith("transformers.generation.continuous_batching"):
                del sys.modules[name]

        _install_transformers_continuous_batching_shim()

        self.assertIn("transformers.generation.continuous_batching", sys.modules)
        self.assertIn("transformers.generation.continuous_batching.cache", sys.modules)

    def test_uses_image_text_loader_for_gemma4(self):
        class Config:
            model_type = "gemma4"
            architectures = ["Gemma4ForConditionalGeneration"]

        class AutoConfig:
            @staticmethod
            def from_pretrained(model):
                return Config()

        causal = object()
        image_text = object()

        self.assertIs(_model_loader("model-id", AutoConfig, causal, image_text), image_text)


if __name__ == "__main__":
    unittest.main()
