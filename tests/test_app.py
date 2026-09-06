"""Exercise the app's inference path with model downloads and GPU calls mocked."""

import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import MagicMock, patch


class TestAppInference(unittest.TestCase):
    def setUp(self):
        # Execute the actual app imports and functions, stopping before UI construction.
        path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(path.read_text())
        cutoff = next(i for i, node in enumerate(tree.body)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "custom_css"
                              for t in node.targets))
        tree.body = tree.body[:cutoff]
        self.torch = MagicMock()
        self.processor = MagicMock()
        self.processor.tokenizer.pad_token_id = 0
        self.processor.tokenizer.eos_token_id = 1
        self.processor.tokenizer.decode.return_value = "FINAL: REFUTES"
        self.processor.return_value.to.return_value.input_ids.shape = (1, 10)
        self.model = MagicMock()
        self.model.parameters.side_effect = lambda: iter([types.SimpleNamespace(device="cuda:0")])
        self.base_loader = MagicMock()
        self.adapter_loader = MagicMock()
        self.adapter_loader.from_pretrained.return_value = self.model
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = MagicMock()
        transformers.AutoProcessor.from_pretrained.return_value = self.processor
        transformers.BitsAndBytesConfig = MagicMock()
        transformers.Gemma4UnifiedForConditionalGeneration = self.base_loader
        spaces = types.ModuleType("spaces")
        spaces.GPU = lambda fn: fn
        peft = types.ModuleType("peft")
        peft.PeftModel = self.adapter_loader
        self.namespace = {"__name__": "app_under_test"}
        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "0"}), patch.dict(
            "sys.modules", {"spaces": spaces, "torch": self.torch,
                            "gradio": MagicMock(), "transformers": transformers, "peft": peft}
        ):
            exec(compile(tree, str(path), "exec"), self.namespace)

    def test_model_load_uses_unified_wrapper_and_caches_success(self):
        load = self.namespace["load_verification_model"]
        self.assertEqual(load(), (self.model, self.processor))
        load()
        self.base_loader.from_pretrained.assert_called_once()
        self.assertEqual(self.base_loader.from_pretrained.call_args.kwargs["device_map"], {"": 0})
        self.adapter_loader.from_pretrained.assert_called_once()

    def test_failed_adapter_load_can_be_retried(self):
        self.adapter_loader.from_pretrained.side_effect = [RuntimeError("download failed"), self.model]
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            self.namespace["load_verification_model"]()
        self.assertIsNone(self.namespace["model"])
        self.assertEqual(self.namespace["load_verification_model"](), (self.model, self.processor))

    def test_verification_applies_chat_template_and_keeps_zero_pad_id(self):
        html, prompt, summary = self.namespace["verify_claim"]("Revenue fell.", "Revenue rose.")
        self.assertIn("Refutes", html)
        self.assertIn("`REFUTES`", summary)
        self.processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        self.assertFalse(self.processor.call_args.kwargs["add_special_tokens"])
        self.assertEqual(self.model.generate.call_args.kwargs["pad_token_id"], 0)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 24)

    def test_parser_uses_last_complete_final_label(self):
        self.processor.tokenizer.decode.return_value = "FINAL: SUPPORTS\nFINAL: REFUTES"
        self.assertIn("`REFUTES`", self.namespace["verify_claim"]("c", "e")[2])
        self.processor.tokenizer.decode.return_value = "FINAL: SUPPORTS_EXTRA <script>"
        html, _, summary = self.namespace["verify_claim"]("c", "e")
        self.assertIn("PARSE_ERROR", summary)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)

    def test_empty_input_does_not_load_model(self):
        self.assertIn("Claim and evidence are required", self.namespace["verify_claim"](" ", "e")[0])
        self.base_loader.from_pretrained.assert_not_called()

    def test_load_error_is_reported_and_escaped(self):
        self.adapter_loader.from_pretrained.side_effect = RuntimeError("<download failed>")
        with patch("builtins.print"):
            html, _, summary = self.namespace["verify_claim"]("c", "e")
        self.assertIn("INFERENCE ERROR", html)
        self.assertIn("&lt;download failed&gt;", html)
        self.assertIn("RuntimeError", summary)


if __name__ == "__main__":
    unittest.main()
