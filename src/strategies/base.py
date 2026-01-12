from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class AlignmentStrategy(ABC):
    """Abstract Base Class for define Assets collecting"""
    @abstractmethod
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Takes a dictionary of {symbol: dataframe} and returns a single DataFrame
        that defines the 'aligned' index structure.
        """
        pass