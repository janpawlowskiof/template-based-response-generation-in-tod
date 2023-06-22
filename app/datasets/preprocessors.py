import re
from typing import Any, Dict, List, Union

from app.utils import merge_dicts, preprocessor_slot_description_to_value


def get_proprocessors(preprocessors_config: List[Any]):
    if preprocessors_config is None:
        return None
    preprocessors = [_get_proprocessor(preprocessor_config) for preprocessor_config in preprocessors_config]

    def combined_preprocessors(x):
        for preprocessor in preprocessors:
            x = preprocessor(x)
        return x

    return combined_preprocessors


def get_metadata_proprocessors(preprocessors_config: List[Any]):
    if preprocessors_config is None:
        return None
    preprocessors = [_get_proprocessor(preprocessor_config) for preprocessor_config in preprocessors_config]

    def combined_preprocessors(x):
        return merge_dicts([
            preprocessor(x)
            for preprocessor in preprocessors
        ])

    return combined_preprocessors


def _get_proprocessor(preprocessor_config: Union[Dict[str, Any], str]):
    if isinstance(preprocessor_config, str):
        preprocessor_name = preprocessor_config
        kwargs = {}
    elif isinstance(preprocessor_config, dict):
        preprocessor_name = preprocessor_config.pop("name")
        kwargs = preprocessor_config
    else:
        raise RuntimeError(f"Unknown preprocessor_config type {type(preprocessor_config)}")
    match preprocessor_name:
        case None:
            return None
        case "slot_description_to_value":
            return preprocessor_slot_description_to_value
        case "slot_description_to_metadata":
            return _preprocessor_slot_description_to_metadata
        case "remove_quotqation":
            return _preprocessor_remove_quotqation
        case _:
            raise ValueError(f"Unknown preprocessor name {preprocessor_name}!")


def _preprocessor_slot_description_to_metadata(x):
    r_groups = re.compile(r"\[service=(?P<service>.*?), slot=(?P<slot>.*?), value=(?P<value>.*?)]")
    return {"used_slots": [m.groupdict() for m in r_groups.finditer(x)]}


def _preprocessor_remove_quotqation(x: str):
    if x.startswith("\"") and x.endswith("\""):
        return x[1:-1]
    return x
