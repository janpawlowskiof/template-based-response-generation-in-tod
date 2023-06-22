import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch.utils.data
from pqdm.threads import pqdm
from transformers import PreTrainedTokenizer


class PicklingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, cache_path: Path, force_recreate_cached_items: bool) -> None:
        super().__init__()
        self.base_dataset: torch.utils.data.Dataset = base_dataset
        self.cache_path: Path = cache_path
        self.force_recreate_cached_items: bool = force_recreate_cached_items
        self.cached_items: List[Any] = []
        self.load_cached_items()

    def __len__(self) -> int:
        return len(self.cached_items)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cached_items[idx]

    def load_cached_items(self):
        if not self.cache_path.exists() or self.force_recreate_cached_items:
            self.recreate_cached_items()
            return
        with self.cache_path.open("rb") as cache_file:
            self.cached_items = pickle.load(cache_file)
            if len(self.cached_items) != len(self.base_dataset):
                print("Loaded dataset was different length compared to loaded dataset")
                self.recreate_cached_items()

    def recreate_cached_items(self):
        print("Recreating dataset cache")
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)
        self.cached_items = pqdm(list(range(len(self.base_dataset))), self.base_dataset.__getitem__, n_jobs=16)
        with self.cache_path.open("wb") as cache_file:
            pickle.dump(self.cached_items, cache_file)
