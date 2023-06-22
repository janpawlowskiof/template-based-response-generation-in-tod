from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class NLGLightningModuleConfig:
    pretrained_model_type: str
    pretrained_model_path: str

    tokenizer_type: str
    tokenizer_path: str

    use_search_in_dev: bool
    use_search_in_test: bool
    num_beams_in_search: int = 4
    num_return_sequences: int = 1
    calculate_metrics_in_train: bool = False
    select_best_response_method: str = "first"
    sacc_name_for_correction_loop: str = "sacc_flan_t5_large"

    pickled_model_path: Path = None

    lr: float = 1e-04

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> NLGLightningModuleConfig:
        return NLGLightningModuleConfig(**config_dict)
