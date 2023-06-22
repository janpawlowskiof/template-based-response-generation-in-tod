import torch.utils.data


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, num_repeats: int) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return len(self.base_dataset) * self.num_repeats

    def __getitem__(self, index):
        local_index = index % len(self.base_dataset)
        return self.base_dataset[local_index]
