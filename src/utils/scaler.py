import torch
import logging
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class SklearnWrapper:
    """
    Wrapper class to use Scikit-Learn scalers with PyTorch Tensors.
    It handles device movement (CPU/GPU) and reshaping automatically.
    """
    def __init__(self, sklearn_scaler: BaseEstimator):
        """ e.g., MinMaxScaler(feature_range=(-1, 1)) """
        self.scaler = sklearn_scaler
        self.fitted = False
        self.n_features = None

    def fit(self, data: torch.Tensor):
        """
        Compute the mean/std/min/max to be used for later scaling.
        Args:
            data: Tensor of shape [B, L, N, F] or [..., F]
        """
        # Check num of Feature
        self.n_features = data.shape[-1]

        # Transform Tensor (GPU/CPU) -> Numpy Array (CPU)
        # Reshape to 2D [Total_Samples, F] because sklearn requires
        flat_data = data.detach().cpu().numpy().reshape(-1, self.n_features)
        
        # let sklearn calc statistic
        self.scaler.fit(flat_data)
        self.fitted = True
        
        logger.debug(f"Scaler Fitted using: {self.scaler.__class__.__name__}")


    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """ Norm data (e.g., Range data -> [-1, 1]) """
        if not self.fitted:
            raise ValueError("Scaler not fitted yet! Call .fit() first.")
            
        original_shape = data.shape
        device = data.device
        
        # Transform to Numpy 2D
        flat_data = data.detach().cpu().numpy().reshape(-1, data.shape[-1])
        
        # Transform
        scaled_data = self.scaler.transform(flat_data)
        
        # Transform back to Tensor on original Device + return original Shape
        scaled_tensor = torch.from_numpy(scaled_data).float().to(device)
        return scaled_tensor.view(original_shape)

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """ DeNorm data (e.g., [-1, 1] -> data) """
        if not self.fitted:
            raise ValueError("Scaler not fitted yet! Call .fit() first.")
            
        original_shape = data.shape
        device = data.device
        
        # 1. Transform Numpy 2D
        flat_data = data.detach().cpu().numpy().reshape(-1, data.shape[-1])
        
        # 2. Inverse Transform value
        original_data = self.scaler.inverse_transform(flat_data)
        
        # 3. Transform back to Tensor on original Device
        original_tensor = torch.from_numpy(original_data).float().to(device)
        return original_tensor.view(original_shape)