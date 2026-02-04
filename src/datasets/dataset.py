import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, List, Tuple
from numpy.lib.stride_tricks import sliding_window_view


class MarketDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int, target_idx: int = 3):
        # [T, A, F]
        # [Time, Assets, Features]
        self.data = data
        self.window_size = window_size
        self.target_idx = target_idx

        # [Num_Win, Assets, Features, Win_Size]
        _windows = sliding_window_view(self.data, window_shape=window_size, axis=0)
        
        # [Num_Win, Win_Size, Asset, Feature]
        # Win_Size means Time (T) so [Num_Win, Time, Assets, Features]
        # From _windows shape = [0(Num_Win), 1(Asset), 2(Feature), 3(Win_Size)]
        self.windows = _windows.transpose(0, 3, 1, 2)
        
    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        window_data = self.windows[idx]

        # x shape: [T, A, 1] 
        x = window_data[:, :, [self.target_idx]]

        n_features = window_data.shape[-1]
        all_indices = np.arange(n_features)
        cond_indices = all_indices[all_indices != self.target_idx]

        # x condition shape: [T, A, F_cond]
        x_cond = window_data[:, :, cond_indices]
        
        return {
            "x": x.astype(np.float32),       
            "x_cond": x_cond.astype(np.float32),
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