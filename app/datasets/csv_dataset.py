import csv
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict
from app.datasets.preprocessors import get_proprocessors

import torch.utils.data

from app.datasets.metadata import Metadata


class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: Path,
        delimiter: str,
        x_col_id: int,
        y_col_id: int,
        x_preprocessors: Optional[List[str]] = None,
        y_preprocessors: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.x_col_id: int = x_col_id
        self.y_col_id: int = y_col_id
        self.x_preprocessor: Optional[Callable[[str], str]] = get_proprocessors(x_preprocessors)
        self.y_preprocessor: Optional[Callable[[str], str]] = get_proprocessors(y_preprocessors)

        with self.csv_path.open("r") as csv_file:
            self.lines: List[List[str]] = list(csv.reader(csv_file, delimiter=delimiter))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> Dict[str, str]:
        line = self.lines[idx]
        x = line[self.x_col_id]
        y = line[self.y_col_id]

        if self.x_preprocessor:
            x = self.x_preprocessor(x)
        if self.y_preprocessor:
            y = self.y_preprocessor(y)

        return {"x": x, "y": y, "metadata": Metadata()}
