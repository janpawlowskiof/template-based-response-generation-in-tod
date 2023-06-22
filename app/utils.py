import os
from pathlib import Path
import re
from typing import Any, Dict, List

import yaml

CACHE_PATH = Path(os.getenv("CACHE_PATH", "/mnt/data/jpawlowski/cache/"))
WANDB_CACHE_PATH = Path(os.getenv("WANDB_CACHE_PATH", "/mnt/data/jpawlowski/wandb/"))
APP_PATH = Path(__file__).parent
ROOT_PATH = APP_PATH.parent
CONFIGS_DIR_PATH = Path(os.getenv("CONFIGS_DIR_PATH", ROOT_PATH / "configs"))
SACC_MODELS_CONFIG_PATH = CONFIGS_DIR_PATH / "sacc_models_config.json"
print(SACC_MODELS_CONFIG_PATH)


def load_config(path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    with Path(path).open() as file:
        cfg = yaml.safe_load(file)
    return cfg


def merge_dicts(dicts: List[Dict]) -> Dict:
    result = {}
    for d in dicts:
        result = {
            **result, **d
        }
    return result


def prepare_correction_input(input_text: str, parts: List[Dict], sacc_results: Dict[str, bool]) -> str:
    templates_that_are_missing = [
        preprocessor_slot_description_to_value(part["template"])
        for part in parts
        if sacc_results[part["slot"]] is False
    ]
    return " ".join([input_text] + templates_that_are_missing)


def preprocessor_slot_description_to_value(x):
    r_groups = re.compile(r"\[service=(?P<service>.*?), slot=(?P<slot>.*?), value=(?P<value>.*?)]")
    slot_format_str = "[service={service}, slot={slot}, value={value}]"

    matched_groups = [m.groupdict() for m in r_groups.finditer(x)]
    for matched_group in matched_groups:
        slot_description = slot_format_str.format(**matched_group)
        x = x.replace(slot_description, matched_group["value"])
    return x
