from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class AlignmentStrategy(ABC):
    """Abstract Base Class for define Assets collecting"""
    @abstractmethod
    def align(self, data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        pass