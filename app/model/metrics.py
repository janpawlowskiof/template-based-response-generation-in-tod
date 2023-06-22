from collections import defaultdict
from typing import Any, Dict, List

import evaluate

from app.rankers.accuracy_ranker import calculate_slots_accuracies

bleu_fn = evaluate.load("bleu")
rouge_fn = evaluate.load("rouge")


def calculate_bleu(*, references: List[str], predictions: List[str], metadata: List[Any] = None):
    try:
        return bleu_fn.compute(references=references, predictions=predictions)["bleu"]
    except ZeroDivisionError:
        return 0.0


def calculate_rouge2(*, references: List[str], predictions: List[str], metadata: List[Any] = None):
    try:
        return rouge_fn.compute(references=references, predictions=predictions, rouge_types=['rouge2'])['rouge2']
    except ZeroDivisionError:
        return 0.0


def batched_calculate_slots_accuracy(*, predictions: List[str], metadata: List[Any], return_list: bool = False, return_internal_dict: bool = False):
    scores = [
        calculate_slots_accuracies(utterance=prediction, metadata=metadata_entry, return_dict=return_internal_dict)
        for prediction, metadata_entry
        in zip(predictions, metadata)
    ]
    if return_list:
        return scores
    scores_t = defaultdict(list)
    for score in scores:
        for score_name, score_value in score.items():
            scores_t[score_name].append(score_value)
    _assert_all_dict_items_have_same_length(scores_t)
    return {
        score_name: sum(score_values) / len(score_values)
        for score_name, score_values in scores_t.items()
    }

def _assert_all_dict_items_have_same_length(d):
    values = list(d.values())
    for value in values:
        assert len(value) == len(values[0])


def calculate_metrics(*, references: List[str] = None, predictions: List[str], metadata: List[Any]) -> Dict[str, float]:
    return {
        "bleu": calculate_bleu(references=references, predictions=predictions, metadata=metadata),
        "rouge2": calculate_rouge2(references=references, predictions=predictions, metadata=metadata),
        **batched_calculate_slots_accuracy(predictions=predictions, metadata=metadata)
    }
