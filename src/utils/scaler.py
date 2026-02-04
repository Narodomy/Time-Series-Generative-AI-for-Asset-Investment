import torch
import logging
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def scale(part: np.ndarray, scaler) -> np.ndarray:
    T, A, F = part.shape
    part_2d = part.reshape(-1, F)

    # print(f"2D Part: {part_2d.shape}")

    scaled_part = scaler.transform(part_2d).reshape(T, A, F)
    return scaled_part.astype(np.float32)

def inverse_scale(scaled_part: np.ndarray, scaler) -> np.ndarray:
    original_shape = scaled_part.shape
    F = original_shape[-1]

    part_2d = scaled_part.reshape(-1, F)
    
    unscaled_2d = scaler.inverse_transform(part_2d)
    return unscaled_2d.reshape(original_shape).astype(np.float32)

def inverse_scale_pair(x, x_cond, scaler):
    if torch.is_tensor(x):      x = x.cpu().numpy()
    if torch.is_tensor(x_cond): x_cond = x_cond.cpu().numpy()
        
    # Check Shape
    assert x.shape[:-1] == x_cond.shape[:-1], "Dimensions mismatch!"
    
    # 2. Concatenate to recreate the feature set used during training
    # [..., 1] + [..., 6] -> [..., 7]
    full_features = np.concatenate([x, x_cond], axis=-1)
    original_shape = full_features.shape
    
    # 3. Inverse Transform
    flat_data = full_features.reshape(-1, original_shape[-1])
    unscaled_flat = scaler.inverse_transform(flat_data)
    unscaled_full = unscaled_flat.reshape(original_shape)
    
    # 4. Split back into x and x_cond
    target_dim = x.shape[-1] # Normal = 1
    
    x_real      = unscaled_full[..., :target_dim]  # (Price)
    x_cond_real = unscaled_full[..., target_dim:]  # (Condition)
    
    return x_real.astype(np.float32), x_cond_real.astype(np.float32)

def inverse_scale_with_cond(x, x_cond, scaler):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(x_cond):
        x_cond = x_cond.cpu().numpy()
        
    assert x.shape[:-1] == x_cond.shape[:-1], f"Shape Mismatch: x {x.shape} vs cond {x_cond.shape}"
    
    # [..., 1] + [..., 6] -> [..., 7]
    full_features = np.concatenate([x, x_cond], axis=-1)

    original_shape = full_features.shape
    total_features = original_shape[-1]

    flat_data = full_features.reshape(-1, total_features)

    unscaled_flat = scaler.inverse_transform(flat_data)

    unscaled_full = unscaled_flat.reshape(original_shape)
    unscaled_price = unscaled_full[..., 0:1]
    return unscaled_price.astype(np.float32)