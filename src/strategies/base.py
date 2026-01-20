from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class AlignmentStrategy(ABC):
    def clean(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cleaned_map = {}
        for symbol, df in data_map.items():
            df_cleaned = df.dropna(how='any')
            
            if not df_cleaned.empty:
                cleaned_map[symbol] = df_cleaned
            else:
                logger.warning(f"Asset '{symbol}' removed: became empty after dropping NaNs.")
    
        return cleaned_map
    
    @abstractmethod
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement .align()")