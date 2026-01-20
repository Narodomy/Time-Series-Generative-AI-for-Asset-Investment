import logging
import pandas as pd
from typing import Dict
from .base import AlignmentStrategy

logger = logging.getLogger(__name__)

class IntersectionStrategy(AlignmentStrategy):
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data_map:
            return pd.DataFrame()

        clean_map = self.clean(data_map)
        
        if not clean_map:
            logger.warning("All assets were empty after cleaning. Returning empty DataFrame.")
            return pd.DataFrame()
        try:
            # keys=data_map.keys() creates a MultiIndex column structure (Symbol, Feature)
            # join='inner' performs the intersection on the Index (Date)
            aligned_df = pd.concat(
                clean_map.values(), 
                axis=1, 
                keys=clean_map.keys(), 
                join='inner'
            )
            
            logger.info(f"Aligned: {len(data_map)} orig -> {len(clean_map)} clean assets -> {len(aligned_df)} rows")
            return aligned_df
            
        except Exception as e:
            logger.error(f"Alignment Error: {e}")
            return pd.DataFrame()

        return aligned_df

class UnionStrategy(AlignmentStrategy):
    """
    [Loose Mode] Outer Join Logic.
    Keeps all timestamps from start to finish.
    
    Use Case:
    - Handling assets with different IPO dates.
    - Note: This will result in NaNs in the Basket. You must handle imputation later!
    """
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data_map:
            return pd.DataFrame()

        aligned_df = pd.concat(
            data_map.values(), 
            axis=1, 
            keys=data_map.keys(), 
            join='outer'
        )
        
        logger.debug(f"Union Strategy: Aligned {len(data_map)} assets. Total rows: {len(aligned_df)}")
        return aligned_df


class DateRangeStrategy(AlignmentStrategy):
    """
    [Fixed Mode] specific start and end date.
    Forces the index to be within a specific range.
    """
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # 1. First, do an outer join to get everything
        df = pd.concat(data_map.values(), axis=1, keys=data_map.keys(), join='outer')
        
        # 2. Slice by date range
        mask = (df.index >= self.start_date) & (df.index <= self.end_date)
        df = df.loc[mask]
        
        logger.debug(f"DateRange Strategy: {self.start_date.date()} to {self.end_date.date()}. Rows: {len(df)}")
        return df