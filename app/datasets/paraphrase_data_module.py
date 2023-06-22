import random
from functools import partial
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.utilities.data import CombinedLoader
from transformers import PreTrainedTokenizer
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from pytorch_lightning.loggers import WandbLogger

from app.datasets.csv_dataset import CSVDataset
from app.datasets.json_dataset import JsonDataset
from app.datasets.metadata import Metadata
from app.datasets.quora_dataset import QuoraDataset
from app.datasets.predictions_dataset import PredictionsDataset
from app.datasets.repeat_dataset import RepeatDataset
from app.datasets.tokenizer_dataset import TokenizerDataset


class ParaphraseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_run: WandbLogger,
        tokenizer: PreTrainedTokenizer,
        ds_split_kwargs: Dict,
        dl_split_kwargs: Dict,
        is_decoder_only: bool,
        max_token_length: int = 256,
        num_workers: int = 16,
        prefetch_factor: int = 2,
        **common_kwargs
    ) -> None:
        super().__init__()
        self.wandb_run: WandbLogger = wandb_run
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.is_decoder_only: bool = is_decoder_only
        self.max_token_length: int = max_token_length
        self.num_workers: int = num_workers
        self.prefetch_factor: int = prefetch_factor
        # split -> ds_name -> kwargs dict
        self.ds_split_kwargs: Dict[str, Dict[str, Dict[str, Any]]] = ds_split_kwargs
        self.dl_split_kwargs: Dict[str, Any] = dl_split_kwargs
        self.common_kwargs: Dict = common_kwargs

    def train_dataloader(self):
        split = "train"
        datasets = self.load_datasets(split=split)
        dataset = torch.utils.data.ConcatDataset(datasets.values())
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=get_collate_with_metadata(),
            **self.dl_split_kwargs[split]
        )

    def val_dataloader(self):
        return self.val_test_dataloader("dev")

    def test_dataloader(self):
        return self.val_test_dataloader("test")

    def val_test_dataloader(self, split: str):
        datasets = self.load_datasets(split=split)
        dataloaders = {
            ds_name: torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=self.prefetch_factor,
                collate_fn=get_collate_with_metadata(),
                **self.dl_split_kwargs[split]
            )
            for ds_name, dataset in datasets.items()
        }
        return CombinedLoader(dataloaders, mode="min_size")

    def predict_dataloader(self):
        return self.test_dataloader()

    def load_datasets(self, split: str) -> Dict[str, torch.utils.data.Dataset]:
        datasets = {
            ds_name: self.load_dataset(split=split, **dataset_kwargs)
            for ds_name, dataset_kwargs in self.ds_split_kwargs[split].items()
        }
        print(f"Loaded datasets for split {split} with lengths:")
        for ds_name, ds in datasets.items():
            print(f"\t{ds_name}: {len(ds)}")
        return datasets

    def load_dataset(
        self, ds_type: str, split: str, subset: Optional[int] = None, num_repeats: Optional[int] = None, **ds_kwargs
    ) -> TokenizerDataset:
        base_dataset_class = {
            "json_dataset": partial(JsonDataset, split=split),
            "csv_dataset": CSVDataset,
            "quora_dataset": QuoraDataset,
            "prediction_dataset": partial(PredictionsDataset, wandb_run=self.wandb_run, split=split),
        }[ds_type]
        base_dataset = base_dataset_class(
            **ds_kwargs,
            **self.common_kwargs
        )
        dataset = TokenizerDataset(
            base_dataset=base_dataset, 
            tokenizer=self.tokenizer, 
            max_token_length=self.max_token_length, 
            is_decoder_only=self.is_decoder_only
        )
        if subset:
            indices = list(range(len(dataset)))
            indices = random.sample(indices, subset)
            dataset = torch.utils.data.Subset(dataset, indices=indices)
        if num_repeats:
            dataset = RepeatDataset(dataset, num_repeats=num_repeats)
        return dataset


def get_collate_with_metadata():
    collate_fn_map = default_collate_fn_map.copy()
    collate_fn_map[Metadata] = Metadata.collate

    def collate_with_metadata(batch):
        return collate(batch, collate_fn_map=collate_fn_map)

    return collate_with_metadata
