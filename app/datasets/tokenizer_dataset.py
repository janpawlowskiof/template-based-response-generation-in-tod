from typing import Dict, Optional, Tuple

import torch.utils.data
from transformers import PreTrainedTokenizer

from app.datasets.metadata import Metadata


class TokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, tokenizer: PreTrainedTokenizer,
                 max_token_length: int, is_decoder_only: bool) -> None:
        super().__init__()
        self.base_dataset: torch.utils.data.Dataset = base_dataset
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_token_length = max_token_length
        self.is_decoder_only: bool = is_decoder_only

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        entry = self.base_dataset[idx].copy()
        assert isinstance(entry["metadata"], Metadata)
        x = entry.pop("x")
        y = entry.pop("y")
        if self.is_decoder_only:
            # input here is inda useless as only targets are used.
            prefix = self.tokenizer.bos_token + x + self.tokenizer.sep_token
            text = prefix + y + self.tokenizer.eos_token
            prefix_tokenized = self.tokenize_text(prefix)
            text_tokenized = self.tokenize_text(text)
            
            input_ids = text_tokenized["input_ids"].clone()
            attention_mask = text_tokenized["attention_mask"]
            target_ids = text_tokenized["input_ids"].clone()
            # masking targets so loss is not calculated on padding
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
            # masking everything until [SEP], including [SEP], loss is not calculated on input
            target_ids[:(target_ids == self.tokenizer.sep_token_id).nonzero()[0] + 1] = -100
            entry["prefix_input_ids"] = prefix_tokenized["input_ids"]
            entry["prefix_attention_mask"] = prefix_tokenized["attention_mask"]
        else:
            x_tokenized = self.tokenize_text(x)
            y_tokenized = self.tokenize_text(y + self.tokenizer.eos_token)
            
            input_ids = x_tokenized["input_ids"]
            attention_mask = x_tokenized["attention_mask"]
            target_ids = y_tokenized["input_ids"]
            # masking targets so loss is not calculated on padding
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            **entry
        }

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }
