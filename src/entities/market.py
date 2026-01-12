import torch
import logging
from typing import Optional, List
from . import Basket
from components import Window

logger = logging.getLogger(__name__)

class Market:
    """
    Market Environment.
    Manages the Assets (Basket) and Time Structure (Window).
    and produces the final [B, L, N, F] tensor.
    """
    def __init__(self, basket: Basket, window: Window):
        self.basket = basket
        self.window = window
        self._batch_tensor: Optional[torch.Tensor] = None # Cache

    def setup(self, features: List[str], device: torch.device = None) -> torch.Tensor:
        """
        Prepares the market data for modeling.
        1. Converts Basket to Tensor [Total_L, N, F]
        2. Applies Window slicing -> [B, L, N, F]
        """
        logger.info("Setting up Market environment")
        
        # Get Raw Data
        # Assumes basket is already aligned (basket.align_assets called)
        raw_tensor = self.basket.to_tensor(features, device)
        
        # Apply Windowing
        self._batch_tensor = self.window.apply(raw_tensor)
        
        logger.info(f"Market Setup Complete. Batch Shape: {self._batch_tensor.shape} [B, L, N, F]")
        return self._batch_tensor

    @property
    def batch(self):
        if self._batch_tensor is None:
            raise RuntimeError("Market not setup! Call market.setup() first.")
        return self._batch_tensor
    
    def __len__(self):
        return self.batch.shape[0] if self._batch_tensor is not None else 0
        