import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import torch.utils.data

from app.datasets.metadata import Metadata


class QuoraDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: Path, override_split: str):
        self.root_path: Path = Path(root_path)
        # TODO: Fix this mess. param cannot be called `split``, because it conflicts with actual split, and we don't want to use the actual split in this class.
        self.override_split: str = override_split

        self.corpus_entries: Dict[str, Dict[str, Any]] = self.load_jsonl_with_regex(regex="*corpus.jsonl")
        self.queries_entries: Dict[str, Dict[str, Any]] = self.load_jsonl_with_regex(regex="*queries.jsonl")

        self.relations: List[Dict[str, str]] = self.load_relations()

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, idx):
        query_index = self.relations[idx]["query-id"]
        corpus_index = self.relations[idx]["corpus-id"]
        x = self.queries_entries[query_index]["text"]
        y = self.corpus_entries[corpus_index]["text"]
        return {"x": x, "y": y, "metadata": Metadata()}

    def load_jsonl_with_regex(self, regex: str) -> Dict[str, Dict[str, Any]]:
        [jsonl_path] = list(self.root_path.glob(regex))
        with jsonl_path.open("r") as jsonl_file:
            jsonl_lines = jsonl_file.readlines()
        entries = [
            json.loads(jsonl_line)
            for jsonl_line in jsonl_lines
        ]
        return {
            entry["_id"]: entry
            for entry in entries
        }

    def load_relations(self) -> List[Dict[str, str]]:
        [tsv_path] = list(self.root_path.glob(f"*{self.override_split}.tsv"))
        with tsv_path.open("r") as tsv_file:
            return list(csv.DictReader(tsv_file, delimiter="\t"))
