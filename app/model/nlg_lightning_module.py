import random
import pickle
from typing import Dict, List, Any, Union
from app.utils import prepare_correction_input

import pytorch_lightning as pl
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from app.model.metrics import batched_calculate_slots_accuracy, calculate_bleu, calculate_metrics, calculate_rouge2
from app.model.nlg_lightning_module_config import NLGLightningModuleConfig
from app.model.pretrained_models import get_pretrained_model, get_tokenizer
from app.rankers.accuracy_ranker import calculate_slots_accuracies

torch.set_float32_matmul_precision('high')


class NLGLightningModule(pl.LightningModule):
    def __init__(self, config: NLGLightningModuleConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config: NLGLightningModuleConfig = config
        self.tokenizer: PreTrainedTokenizer = get_tokenizer(
            tokenizer_type=self.config.tokenizer_type, tokenizer_path=self.config.tokenizer_path,
        )
        if self.config.pickled_model_path:
            # TODO: Chagne or remove this, please.
            print("Loading a pickled model as a base model. TODO: Change or remove this.")
            with self.config.pickled_model_path.open("rb") as pickled_model_file:
                self.model: PreTrainedModel = pickle.load(pickled_model_file)
        else:
            self.model: PreTrainedModel = get_pretrained_model(
                model_type=self.config.pretrained_model_type, model_path=self.config.pretrained_model_path
            )
        if self.is_decoder_only:
            print("Using decoder only model, thus resizing token embeddings.")
            self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def is_decoder_only(self) -> bool:
        return self.config.pretrained_model_type in ["gpt", "mgpt"]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_ids: torch.Tensor
    ) -> Union[Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions]:
        """
        TODO: Add docstring here!
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, metadata: List[Dict]
    ) -> torch.LongTensor:
        """
        TODO: Add docstring here!
        """
        self.config.num_return_sequences = 1
        self.config.num_beams_in_search = 4
        bs = input_ids.shape[0]
        prediction_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            num_beams=self.config.num_beams_in_search,
            num_return_sequences=self.config.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        ).reshape([bs, self.config.num_return_sequences, -1])
        return self.select_best_response(prediction_ids=prediction_ids, metadata=metadata)

    def select_best_response(self, prediction_ids: torch.Tensor, metadata: List[Any]) -> torch.Tensor:
        """
        :param prediction_ids: torch.Tensor of shape [bs, num_sampled_sentences, sentences_length]
        :param metadata: List of metadata for each entry.
        :return torch.Tensor of shape [bs, sentences_length]
        """
        match self.config.select_best_response_method:
            case "first":
                return prediction_ids[:, 0, :]
            case "best_slot_accuracy":
                results = []
                for entry_prediction_ids, entry_metadata in zip(prediction_ids, metadata):
                    if "used_slots" not in entry_metadata:
                        return prediction_ids[:, 0, :]
                    entry_decoded_texts = self.batch_decode(entry_prediction_ids, skip_special_tokens=True, remove_input=True)
                    scores = {
                        entry_single_prediction_ids: calculate_slots_accuracies(entry_decoded_single_text, used_slots=entry_metadata["used_slots"])
                        for entry_single_prediction_ids, entry_decoded_single_text in zip(entry_prediction_ids, entry_decoded_texts)
                    }
                    results.append(max(scores, key=scores.get))
                return torch.stack(results)
            case "random":
                num_sampled = prediction_ids.shape[1]
                return prediction_ids[:, random.randrange(num_sampled), :]
            case "return_all":
                return prediction_ids
            case _:
                raise RuntimeError(f"Unknown response secleciton {self.config.select_best_response_method}")

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        metadata = batch["metadata"]
        # TODO: fix typing
        model_output: Union[Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions] = self.forward(input_ids, attention_mask, target_ids)

        if self.config.calculate_metrics_in_train:
            metrics: Dict[str, torch.Tensor] = self.calculate_metrics_from_model_output(
                target_ids=target_ids, model_output=model_output, metadata=metadata, prefix="train"
            )
        else:
            metrics: Dict[str, torch.Tensor] = {"train_loss": model_output.loss}
        self.log_dict(metrics)

        # it is required by pytorch lightning to have key "loss"
        metrics["loss"] = metrics.pop("train_loss")
        return metrics

    def validation_step(self, batch, batch_idx):
        return self._val_test_step(batch, split="dev", use_search=self.config.use_search_in_dev)

    def test_step(self, batch, batch_idx):
        return self._val_test_step(batch, split="test", use_search=self.config.use_search_in_test)

    def _val_test_step(self, batch, split: str, use_search: bool):
        assert isinstance(batch, dict), "batch should be a dictionary of datasets for validation and test"
        metrics: Dict[str, torch.Tensor] = {}

        for dl_name, dl_batch in batch.items():
            input_ids = dl_batch["input_ids"]
            attention_mask = dl_batch["attention_mask"]
            target_ids = dl_batch["target_ids"]
            metadata = dl_batch["metadata"]
            prefix = f"{split}_{dl_name}"
            dl_metrics: Dict[str, torch.Tensor] = {}
            # TODO: refactor as this is getting slightly messy
            if use_search:
                prediction_ids: torch.Tensor = self.generate(input_ids, attention_mask, metadata=metadata)
            else:
                model_output: Union[Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions] = self.forward(input_ids, attention_mask, target_ids)
                prediction_ids: torch.Tensor = model_output.logits.argmax(dim=-1)
                dl_metrics[f"{prefix}_loss"] = model_output.loss

            dl_metrics.update(
                self.calculate_metrics_from_ids(
                    target_ids=target_ids, prediction_ids=prediction_ids, metadata=metadata, prefix=prefix
                )
            )
            metrics.update(dl_metrics)

        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        assert isinstance(batch, dict)
        results: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        for dl_name, dl_batch in batch.items():
            if self.is_decoder_only:
                input_ids = dl_batch["prefix_input_ids"]
                attention_mask = dl_batch["prefix_attention_mask"]
            else:
                input_ids = dl_batch["input_ids"]
                attention_mask = dl_batch["attention_mask"]
            target_ids = dl_batch["target_ids"]
            metadata = dl_batch["metadata"]
            prediction_ids: torch.LongTensor = self.generate(input_ids, attention_mask, metadata=metadata)
            assert isinstance(prediction_ids, torch.Tensor), f"Expected prediction ids to be torch.LongTensor, but found {type(prediction_ids)}"
            input_texts = self.batch_decode(sequences=input_ids, skip_special_tokens=True, remove_input=False)
            target_texts = self.batch_decode(sequences=target_ids, skip_special_tokens=True, remove_input=True)
            prediction_texts = self.batch_decode(sequences=prediction_ids, skip_special_tokens=True, remove_input=True)
            slot_accuracies: List[Dict[str, Dict[str, bool]]] = batched_calculate_slots_accuracy(
                predictions=prediction_texts, metadata=metadata, return_list=True, return_internal_dict=True
            )

            final_outputs = [
                {
                    "pass1_results": {
                        "model_input": input_text,
                        "text": prediction_text,
                        "sacc": entry_slot_accuracies,
                        "rouge2": calculate_rouge2(references=[target_text], predictions=[prediction_text]),
                        "bleu": calculate_bleu(references=[target_text], predictions=[prediction_text]),
                    }
                }
                for input_text, prediction_text, target_text, entry_slot_accuracies 
                in zip(input_texts, prediction_texts, target_texts, slot_accuracies, strict=True)
            ]
            if self.config.sacc_name_for_correction_loop:
                pass2_results = [
                    self.correction_pass(input_text, entry_metadata=entry_metadata, sacc_results=entry_slot_accuracies[self.config.sacc_name_for_correction_loop])
                    for input_text, entry_metadata, entry_slot_accuracies
                    in zip(input_texts, metadata, slot_accuracies)
                ]
                for pass2_result, target_text in zip(pass2_results, target_texts, strict=True):
                    if not pass2_result:
                        continue
                    pass2_result["rouge2"] = calculate_rouge2(references=[target_text], predictions=[pass2_result["text"]]),
                    pass2_result["bleu"] = calculate_bleu(references=[target_text], predictions=[pass2_result["text"]]),
                final_outputs = [
                    {
                        **final_output,
                        "pass2_results": pass2_result
                    }
                    for final_output, pass2_result 
                    in zip(final_outputs, pass2_results)
                ]

            results[dl_name] = [
                {
                    "input": input_text,
                    "target": target_text,
                    **entry_metadata,
                    **final_output
                }
                for input_text, target_text, entry_metadata, entry_slot_accuracies, final_output
                in zip(input_texts, target_texts, metadata, slot_accuracies, final_outputs, strict=True)
            ]
        return results

    def correction_pass(self, input_text: str, entry_metadata: Dict, sacc_results: Dict[str, bool]) -> Union[Dict[str, Any], None]:
        correction_input_text = prepare_correction_input(input_text, parts=entry_metadata["parts"], sacc_results=sacc_results)
        if correction_input_text == input_text:
            return None
        if self.is_decoder_only:
            correction_input_text = self.tokenizer.bos_token + correction_input_text + self.tokenizer.sep_token
        tokenized_input = self.tokenizer(
            correction_input_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        [correction_ids] = self.generate(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"], metadata=[entry_metadata])
        prediction_text = self.decode(correction_ids, remove_input=True, skip_special_tokens=True)
        slots_accuracies = calculate_slots_accuracies(prediction_text, metadata=entry_metadata, return_dict=True)
        return {
            "model_input": correction_input_text,
            "text": prediction_text,
            "sacc": slots_accuracies
        }

    def calculate_metrics_from_model_output(
        self, target_ids: List[torch.Tensor], model_output: Seq2SeqLMOutput, metadata: List[Any], prefix: str
    ) -> Dict[str, torch.Tensor]:
        prediction_ids: torch.Tensor = model_output.logits.argmax(dim=-1)
        metrics = self.calculate_metrics_from_ids(
            target_ids=target_ids, prediction_ids=prediction_ids, metadata=metadata, prefix=prefix
        )
        metrics[f"{prefix}_loss"] = model_output.loss
        return metrics

    def calculate_metrics_from_ids(
        self, target_ids: List[torch.Tensor], prediction_ids: List[torch.Tensor], metadata: List[Any], prefix: str
    ) -> Dict[str, torch.Tensor]:
        targets_texts = self.batch_decode(sequences=target_ids, skip_special_tokens=True, remove_input=True)
        predictions_texts = self.batch_decode(sequences=prediction_ids, skip_special_tokens=True, remove_input=True)
        metrics = calculate_metrics(references=targets_texts, predictions=predictions_texts, metadata=metadata)
        metrics = {
            f"{prefix}_{metric_name}": metric_value
            for metric_name, metric_value in metrics.items()
        }
        metrics = {
            k: v for k, v in metrics.items() if v
        }
        return metrics

    def decode(self, sequence, skip_special_tokens: bool = True, remove_input: bool = True) -> List[str]:
        sequence[sequence == -100] = self.tokenizer.pad_token_id
        if remove_input and self.is_decoder_only:
            sequence = self._remove_input_from_sequence(sequence)
            sequence = self._remove_text_past_eos(sequence)
        return self.tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = True, remove_input: bool = True) -> List[str]:
        # this needs to be done or decoding text crashes
        sequences[sequences == -100] = self.tokenizer.pad_token_id
        if remove_input and self.is_decoder_only:
            sequences = [
                self._remove_input_from_sequence(sequence)
                for sequence in sequences
            ]
            sequences = [
                self._remove_text_past_eos(sequence)
                for sequence in sequences
            ]
        return self.tokenizer.batch_decode(sequences=sequences, skip_special_tokens=skip_special_tokens)

    def _remove_input_from_sequence(self, sequence) -> torch.Tensor:
        sep_token_indexes = (sequence == self.tokenizer.sep_token_id).nonzero()
        if torch.numel(sep_token_indexes) == 0:
            return sequence
        return sequence[sep_token_indexes[0] + 1:]

    def _remove_text_past_eos(self, sequence) -> torch.Tensor:
        eos_token_indexes = (sequence == self.tokenizer.eos_token_id).nonzero()
        if torch.numel(eos_token_indexes) == 0:
            return sequence
        return sequence[:eos_token_indexes[0] + 1]

    def configure_optimizers(self):
        print("Creating new optimizer!")
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
