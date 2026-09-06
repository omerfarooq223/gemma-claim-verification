"""Loader contracts that do not need model weights or a GPU."""

import typing
import unittest
from unittest.mock import MagicMock, patch
from gemma_claim_verification import modeling


class TestModeling(unittest.TestCase):
    def test_loader_type_annotations_resolve(self):
        self.assertIn("return", typing.get_type_hints(modeling.load_base_model_and_processor))

    def test_hub_adapter_id_reaches_peft(self):
        peft = MagicMock()
        with patch.object(modeling, "PeftModel", peft, create=True):
            result = modeling.load_adapter("base", "owner/adapter")
        peft.from_pretrained.assert_called_once_with("base", "owner/adapter", is_trainable=False)
        self.assertIs(result, peft.from_pretrained.return_value)


if __name__ == "__main__":
    unittest.main()
