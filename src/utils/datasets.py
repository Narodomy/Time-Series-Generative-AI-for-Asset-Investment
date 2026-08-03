"""
utils/datasets.py
──────────────────
torch Dataset/DataLoader wrapper around the saved window tensors.
Kept in its own module (not redefined per-notebook) so any notebook that
loads data back via utils.io.load_data() can reconstruct the exact same
Dataset class without copy-pasting the definition.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    def __init__(self, windows_lr, windows_cond, init_prices, dates, tickers, features):
        self.x = torch.from_numpy(windows_lr).float()
        self.cond = torch.from_numpy(windows_cond).float()
        self.init_price = torch.from_numpy(init_prices).float()
        self.dates = dates  # numpy datetime64 (N,) — ไม่ต้อง to tensor
        self.tickers = tickers
        self.features = features

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],  # (W, A, C_x)
            "cond": self.cond[idx],  # (W, A, C_c)
            "init_price": self.init_price[idx],  # (A, C_x)
            "date_idx": idx,  # lookup ds.dates[idx] ตอน eval
        }


def make_dataloaders(datasets: dict, batch_sizes: dict, num_workers: int = 0,
                      pin_memory: bool = True) -> dict:
    """
    datasets    : {"train": WindowDataset, "val": WindowDataset, "test": WindowDataset}
    batch_sizes : {"train": int, "val": int, "test": int}
    Returns       {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_sizes[split],
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders
