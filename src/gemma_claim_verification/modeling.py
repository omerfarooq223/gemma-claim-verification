"""Model loading, 4-bit NF4 quantization, and LoRA adapter integration."""

import os
from typing import Any, Dict, List, Optional, Union
from .constants import DEFAULT_BASE_MODEL, TARGET_MODULES

try:
    import torch
    import transformers
    from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_bnb_4bit_config() -> Any:
    """Build BitsAndBytesConfig for 4-bit NF4 with double quant and FP16 compute."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch, Transformers, and BitsAndBytes must be installed.")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base_model_and_processor(
    model_name_or_path: str = DEFAULT_BASE_MODEL,
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
    load_in_4bit: bool = True,
) -> Tuple[Any, Any]:
    """Load base Gemma model with 4-bit quantization and processor."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and Transformers must be installed.")

    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"

    quantization_config = get_bnb_4bit_config() if load_in_4bit else None

    # Gemma 4 unified processor
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    # Attempt to load specialized Gemma 4 class if available, else standard CausalLM
    try:
        from transformers import Gemma4UnifiedForConditionalGeneration
        model_cls = Gemma4UnifiedForConditionalGeneration
    except (ImportError, AttributeError):
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    return model, processor


def build_lora_model(
    base_model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Any:
    """Wrap frozen base model with trainable LoRA adapter."""
    if target_modules is None:
        target_modules = TARGET_MODULES

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_adapter(
    base_model: Any,
    adapter_path: str,
    is_trainable: bool = False,
) -> Any:
    """Attach frozen or fine-tuned LoRA adapter to base model."""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    )
    return model
