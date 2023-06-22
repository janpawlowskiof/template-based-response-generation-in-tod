import json
from pathlib import Path
import random
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch.utils.data

from app.datasets.metadata import Metadata
from app.datasets.preprocessors import get_proprocessors, get_metadata_proprocessors


class JsonDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: Path,
        split: str,
        use_random_x: bool,
        x_name: str,
        y_name: str,
        x_preprocessors: List[str] = None,
        y_preprocessors: List[str] = None,
        metadata_preprocessors: List[str] = None,
        required_num_sentences: Optional[List[int]] = None,
        override_split: Optional[str] = None
    ) -> None:
        super().__init__()

        self.json_path: Path = Path(json_path)
        self.split: str = override_split or split
        self.use_random_x: bool = use_random_x
        self.x_name: str = x_name
        self.y_name: str = y_name
        self.x_preprocessor: Optional[Callable[[str], str]] = get_proprocessors(x_preprocessors)
        self.y_preprocessor: Optional[Callable[[str], str]] = get_proprocessors(y_preprocessors)
        self.metadata_preprocessor: Optional[Callable[[str], Dict[str, Any]]] = get_metadata_proprocessors(metadata_preprocessors)
        self.required_num_sentences: Optional[List[int]] = required_num_sentences

        self.xys: List[Tuple[str, str, Dict]] = []
        self.load_entries()
        self.filter_entries()

    def load_entries(self):
        for entry in self.load_json():
            if self.split != "ignore" and entry["split"] != self.split:
                continue
            xs = entry.pop(self.x_name)
            y = entry.pop(self.y_name)
            metadata = entry

            if isinstance(xs, list):
                if self.use_random_x:
                    x = xs[-1]
                else:
                    x = xs[0]
                self.xys.append((x, y, metadata))
            else:
                assert self.use_random_x is False, "When field for x is not a list, you must use use_random_x=False"
                self.xys.append((xs, y, metadata))

    def load_json(self):
        with self.json_path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)

    def filter_entries(self):
        filtered_xys = []
        for x, y, metadata in self.xys:
            num_sentences = len(metadata["parts"])
            if self.required_num_sentences and num_sentences in self.required_num_sentences:
                filtered_xys.append((x, y, metadata))
        self.xys = filtered_xys

    def __len__(self):
        return len(self.xys)

    def __getitem__(self, idx) -> Dict[str, str]:
        x, y, metadata = self.xys[idx]

        metadata: Dict[str, Any] = {
            **metadata,
            **self.metadata_preprocessor(x),
        }
        if self.x_preprocessor:
            x = self.x_preprocessor(x)
        if self.y_preprocessor:
            y = self.y_preprocessor(y)

        return {"x": x, "y": y, "metadata": Metadata(metadata)}
