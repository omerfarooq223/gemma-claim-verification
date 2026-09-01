"""Constants and configurations for Gemma 4 Claim Verification."""

from enum import Enum
from typing import Dict, List, Set

class LabelEnum(str, Enum):
    """Canonical three-way verification labels."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

ALLOWED_LABELS: List[str] = [
    LabelEnum.SUPPORTS.value,
    LabelEnum.REFUTES.value,
    LabelEnum.NOT_ENOUGH_INFO.value,
]

ALLOWED_LABELS_SET: Set[str] = set(ALLOWED_LABELS)

# Label alias canonicalization map
LABEL_ALIAS_MAP: Dict[str, str] = {
    "SUPPORTS": LabelEnum.SUPPORTS.value,
    "supports": LabelEnum.SUPPORTS.value,
    "Supports": LabelEnum.SUPPORTS.value,
    "SUPPORT": LabelEnum.SUPPORTS.value,
    "support": LabelEnum.SUPPORTS.value,
    "SUPPORTED": LabelEnum.SUPPORTS.value,
    "supported": LabelEnum.SUPPORTS.value,
    "REFUTES": LabelEnum.REFUTES.value,
    "refutes": LabelEnum.REFUTES.value,
    "Refutes": LabelEnum.REFUTES.value,
    "REFUTE": LabelEnum.REFUTES.value,
    "refute": LabelEnum.REFUTES.value,
    "REFUTED": LabelEnum.REFUTES.value,
    "refuted": LabelEnum.REFUTES.value,
    "NOT_ENOUGH_INFO": LabelEnum.NOT_ENOUGH_INFO.value,
    "not_enough_info": LabelEnum.NOT_ENOUGH_INFO.value,
    "NOT ENOUGH INFO": LabelEnum.NOT_ENOUGH_INFO.value,
    "not enough info": LabelEnum.NOT_ENOUGH_INFO.value,
    "Not Enough Info": LabelEnum.NOT_ENOUGH_INFO.value,
    "NEI": LabelEnum.NOT_ENOUGH_INFO.value,
    "nei": LabelEnum.NOT_ENOUGH_INFO.value,
    "INSUFFICIENT": LabelEnum.NOT_ENOUGH_INFO.value,
    "insufficient": LabelEnum.NOT_ENOUGH_INFO.value,
}

# Base model and tokenizer identifiers
DEFAULT_BASE_MODEL: str = "google/gemma-4-12B-it"
DEFAULT_MAX_SEQ_LENGTH: int = 256
DEFAULT_MAX_NEW_TOKENS: int = 24

# Target modules for LoRA parameter adaptation
TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Exact prompt formulation used in the winning competition system
SYSTEM_INSTRUCTION: str = (
    "Classify the claim using only the supplied evidence.\n\n"
    "SUPPORTS: the evidence establishes the claim.\n"
    "REFUTES: the evidence contradicts the claim.\n"
    "NOT_ENOUGH_INFO: the evidence neither establishes nor contradicts the specific claim.\n\n"
    "End your response exactly as:\n"
    "FINAL: SUPPORTS\n"
    "or\n"
    "FINAL: REFUTES\n"
    "or\n"
    "FINAL: NOT_ENOUGH_INFO"
)

PROMPT_TEMPLATE: str = (
    "{instruction}\n\n"
    "Claim:\n{claim}\n\n"
    "Evidence:\n{evidence}"
)

# Known artifact hashes for provenance and regression checks
KNOWN_HASHES: Dict[str, str] = {
    # Official Competition Data
    "raw_train1000": "26ea9a6998815d0f99f45aff2206f435781cc71bd92638101d7ea08c2e175d3c",
    "organizer_val300": "722a5f111693996369a02d26eda08f00cdfb51f5b8b842addb752814f03716c9",
    "event_test500": "8811c713e4297c0ce6e89a020d212621af538c74fa3866f4b5d98914ca3b8a28",
    "gold_test500": "696595096e2202422c4396f830e08b4ad1489b59054d2a496271a4ef94cfcbd7",
    
    # Audited & Derived Datasets
    "clean_train935": "b706dbf0c0b4aab4cbcd07bb89c5d018f7c41e47a55b757016ef3d07a9713337",
    "contrastive_train150": "17258cbc0e40f4ebd1cd4d583e3a331e59217d62cf4ff36741e8b1e3a7a98f41",
    "frozen_holdout75": "61f41a537f738ecd153474565c1e22bb678c4f8a9dd096ed9eabfe5bddc2e1b2",
    "external_stress30": "05624200845a79228ac4e05540b013ea0836d049802fcd3f9dcc60ada8aecedc",
    "old_synthetic180": "9b0aad0c39b119e47a91e8c901c5372e0b5e70263606988cfb6f6eee6ba75f52",
    "blind120_input": "67514fdda49707d78a5d52cbf1b8f06d6666f0724b938b59168edd2984f8b030",
    "blind120_gold": "79a3f86d67937b307aeb22de8e6189d69051f28c01c12e4816821f1448124e32",
    
    # Model LoRA Adapters
    "selected_final_adapter": "76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e",
    "d4_champion_adapter": "03b134b92e46f44b3802bf563668436250078a2ffef175e35c0c3c5b5ccf42d7",
    "d1_champion_adapter": "2c64a87e227e0638be9dc73e021f877f2536876ccf94cc116e04d60075164f8f",
    "reproduction_adapter": "cf24e9701752c7dd6d1f35fc9ceaf933f3cdc2dc8e31b5823ba98c444f6b51a9",
    
    # Final Output Submission
    "final_submission_csv": "d4df310da9eb95205daf6bc9ad1f8cc874f7afe5f6d432ce8373f0a9f88dd012",
}
