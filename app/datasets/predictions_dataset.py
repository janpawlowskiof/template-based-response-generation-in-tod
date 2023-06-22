import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import wandb
from app.utils import WANDB_CACHE_PATH, prepare_correction_input

from pytorch_lightning.loggers import WandbLogger


from app.datasets.json_dataset import JsonDataset
from app.datasets.preprocessors import preprocessor_slot_description_to_value


class PredictionsDataset(JsonDataset):
    def __init__(self, wandb_run: WandbLogger, artifact_path: str, sacc_name: str, split: str, x_preprocessors: List[str] = None, y_preprocessors: List[str] = None, metadata_preprocessors: List[str] = None, override_split: str | None = None) -> None:
        json_path = self.download_artifact(wandb_run, artifact_path)
        self.sacc_name = sacc_name
        super().__init__(json_path, split, True, "prediction", "target", x_preprocessors, y_preprocessors, metadata_preprocessors, None, override_split)

    def load_json(self):
        with self.json_path.open("r", encoding="utf-8") as json_file:
            entries = json.load(json_file)
        assert len(entries) == 1
        [key] = list(entries.keys())
        return entries[key]
    
    def filter_entries(self):
        filtered_xys = []
        for x, y, metadata in self.xys:
            if x == y:
                continue
            is_any_slot_missing = not all(metadata["pass1_results"]["sacc"][self.sacc_name].values())
            if is_any_slot_missing:
                filtered_xys.append((x, y, metadata))
        self.xys = filtered_xys

    def download_artifact(self, wandb_run: WandbLogger, artifact_path: str):
        if isinstance(wandb_run, WandbLogger):
            artifact_dir = wandb_run.use_artifact(artifact_path).download(root=WANDB_CACHE_PATH)
        elif isinstance(wandb_run, wandb.Api):
            artifact_dir = wandb_run.artifact(artifact_path).download(root=WANDB_CACHE_PATH)
        else:
            raise ValueError(f"Unknown wandb run type {type(wandb_run)}")
        json_path = Path(artifact_dir) / "predictions.json"
        return json_path

    def __getitem__(self, idx) -> Dict[str, str]:
        entry = super().__getitem__(idx)
        entry["x"] = prepare_correction_input(
            entry["x"], parts=entry["metadata"]["parts"], sacc_results=entry["metadata"]["pass1_results"]["sacc"][self.sacc_name]
        )
        return entry

    def __len__(self):
        return super().__len__()
