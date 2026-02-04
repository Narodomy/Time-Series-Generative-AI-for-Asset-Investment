import torch
import numpy as np

def inspect(data, name: str = ""):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    _min = np.min(data)
    _max = np.max(data)
    _mean = np.mean(data)
    _std = np.std(data)
    
    print(f"--- Inspecting: {name} ---")
    print("-" * 36)
    print(f"Shape: {data.shape}")
    print(f"Min:   {_min:.4f}")
    print(f"Max:   {_max:.4f}")
    print(f"Mean:  {_mean:.4f}")
    print(f"Std:   {_std:.4f}")
    print("-" * 36)
    return _min, _max, _mean, _std


def inverse_log_returns(r_log: np.ndarray) -> np.ndarray:
    # R_simple = e^(R_log) - 1
    r_simple = np.exp(r_log) - 1

    return r_simple