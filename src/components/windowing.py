import torch
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Window(ABC):
    """
    Base class for windowing logic.
    Defines how we slice the timeline into batches.
    """
    def __init__(self, size: int):
        self.size = size

    @abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """
        Input: [Total_Len, N, F]
        Output: [Batch, Size, N, F]
        """
        pass

class RollingWindow(Window):
    """
    Standard sliding window (Moving window).
    """
    def __init__(self, size: int, stride: int = 1):
        super().__init__(size)
        self.stride = stride

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        L_total, N, F = data.shape
        
        if L_total < self.size:
            raise ValueError(f"Data length ({L_total}) < Window size ({self.size})")

        # 1. Unfold (Slide)
        # data.unfold(dim, size, step) -> [Num_Windows, N, F, Window_Size]
        windows = data.unfold(0, self.size, self.stride)
        
        # 2. Permute to [Batch, Window_Size, N, F]
        # [B, N, F, L] -> [B, L, N, F]
        windows = windows.permute(0, 3, 1, 2)
        
        logger.debug(f"RollingWindow applied. Result: {windows.shape}")
        return windows