import re
from app.utils import load_config, SACC_MODELS_CONFIG_PATH
import torch
import spacy

from typing import Any, Dict, List, Union

from app.rankers.flan_t5_ranker import FlanT5Sacc
from app.rankers.flan_t5_ranker_pl import FlanT5SaccPL


lemmatize_nlp = None
sentiment_nlp = None


multipliers = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}


def load_sacc_models(names: List[str], device) -> Dict[str, Any]:
    def _load_sacc_model(name):
        match name:
            case "sacc_flan_t5_large":
                return FlanT5Sacc(size="large", device=device)
            case "sacc_flan_t5_large_pl":
                return FlanT5SaccPL(size="large", device=device)
            case _:
                raise RuntimeError(f"Unknown name {name}")
    return {
        sacc_name: _load_sacc_model(sacc_name)
        for sacc_name in names
    }
            

SACC_MODELS = load_sacc_models(names=[], device="cuda:0")


def calculate_slots_accuracies(
    utterance: str, metadata: Dict, default_value: float = 1.0, return_dict: bool = False,
) -> Dict[str, Union[float, Dict[str, bool]]]:
    results = {
        "sacc_literal": calculate_literal_matching_slots_accuracy(utterance, parts=metadata.get("parts", []), lower=True, lemma=False),
        **{
            flan_model_name: flan_model.calculate(utterance=utterance, metadata=metadata) 
            for flan_model_name, flan_model 
            in SACC_MODELS.items()
        }
    }
    if return_dict:
        return results
    return {
        k: sum(v.values()) / len(v) if v else default_value
        for k, v
        in results.items()
    }


def calculate_literal_matching_slots_accuracy(
    utterance: str, parts: List[Dict[str, Any]], lower: bool = True, lemma: bool = False
) -> Dict[str, bool]:
    return {
        part["slot"]: all(
            is_slot_value_used(utterance, slot_value, lower=lower, lemma=lemma)
            for slot_value in part["values"]
        )
        for part in parts
    }


def is_slot_value_used(utterance: str, slot_value: str, lower: bool, lemma: bool) -> bool:
    utterance = remove_punctuation(utterance)
    slot_value = remove_punctuation(slot_value)

    if lower:
        utterance = utterance.lower()
        slot_value = slot_value.lower()

    if lemma:
        lemma_utterance = surround_with_spaces(lemmatize(utterance))
        lemma_slot_value = surround_with_spaces(lemmatize(slot_value))
        if lemma_slot_value in lemma_utterance:
            return True

    utterance = surround_with_spaces(utterance)
    slot_value = surround_with_spaces(slot_value)
    return slot_value in utterance


def remove_punctuation(text: str) -> str:
    text = text.replace("in the morning", "am")
    text = text.replace("in the evening", "pm")

    text = text.replace("-", " ")
    return re.sub(r'[^\w\s]', '', text)


def surround_with_spaces(text):
    return f" {text} "


def lemmatize(text):
    global lemmatize_nlp
    if not lemmatize_nlp:
        lemmatize_nlp = spacy.load("pl_core_news_sm")
    return " ".join([t.lemma_ for t in lemmatize_nlp(text) if t.lemma_])
