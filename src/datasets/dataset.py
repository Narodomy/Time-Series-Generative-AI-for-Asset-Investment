import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, List, Tuple

class MarketDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, mode: str = 'exhaustive', device: torch.device = None):
        """
        Args:
            data_tensor: [Total_Windows, L, N, F] -> [2660, 64, 10, 2]
            mode: 'exhaustive' (Total = 2660*10) or 'random' (Total = 2660)
        """
        self.data = data_tensor
        self.mode = mode
        
        if device:
            self.data = self.data.to(device)
            
        self.n_windows = self.data.shape[0]
        self.window_size = self.data.shape[1]
        self.n_assets = self.data.shape[2]
        self.n_features = self.data.shape[3]

    def __len__(self):
        if self.mode == 'exhaustive':
            return self.n_windows * self.n_assets # N * (N assets)
        else:
            return self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'exhaustive':
            window_idx = idx // self.n_assets
            target_asset_idx = idx % self.n_assets
        else:
            window_idx = idx
            target_asset_idx = torch.randint(0, self.n_assets, (1,)).item()

        window_sample = self.data[window_idx] # Shape: [64, 10, 2]

        # Target 1 asset
        target_data = window_sample[:, target_asset_idx, :] # Shape: [64, 2]
        
        # Context N - 1 assets
        all_indices = torch.arange(self.n_assets)
        context_mask = (all_indices != target_asset_idx)
        context_data = window_sample[:, context_mask, :] # Shape: [64, 9, 2]

        return {
            "target": target_data,       
            "context": context_data,     
            "target_idx": torch.tensor(target_asset_idx),
            "window_idx": torch.tensor(window_idx)
        }


class JointMarketDataset(Dataset):
    def __init__(self, data_tensor):
        """
        data_tensor: [Batch (All Time), N, F], [All Time, N, F]
        """
        self.data = data_tensor
        self.n_windows = self.data.shape[0]
    
    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        return self.data[idx]

def create_randomize_datasets(
    dataset: Dataset, 
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    
    assert sum(split_ratios) == 1.0
    
    # Check dataset type to determine total windows
    if hasattr(dataset, 'n_windows'):
        total_windows = dataset.n_windows
    else:
        # Fallback if manual dataset
        total_windows = len(dataset) 
        
    all_window_indices = np.arange(total_windows)
    
    # Shuffle Time Windows (Not samples)
    np.random.seed(seed)
    np.random.shuffle(all_window_indices)
    
    val_size = int(total_windows * split_ratios[1])
    test_size = int(total_windows * split_ratios[2])
    
    val_window_indices = all_window_indices[:val_size]
    test_window_indices = all_window_indices[val_size : val_size + test_size]
    
    # Train takes the rest and sorts them (Time integrity within train set)
    train_window_indices = all_window_indices[val_size + test_size:]
    train_window_indices = np.sort(train_window_indices)
    
    # Helper: Expand indices for Exhaustive Mode
    def expand_indices(window_indices):
        # Case 1: MarketDataset with 'exhaustive' mode
        if hasattr(dataset, 'mode') and dataset.mode == 'exhaustive':
            final_indices = []
            for w in window_indices:
                start = w * dataset.n_assets
                end = start + dataset.n_assets
                final_indices.extend(range(start, end))
            return final_indices
            
        # Case 2: MarketDataset(random) OR JointMarketDataset
        else:
            return window_indices.tolist()

    return (
        Subset(dataset, expand_indices(train_window_indices)),
        Subset(dataset, expand_indices(val_window_indices)),
        Subset(dataset, expand_indices(test_window_indices))
    )