import logging
import pandas as pd
from typing import Dict
from .base import AlignmentStrategy

logger = logging.getLogger(__name__)

class StrictAlignment(AlignmentStrategy):
    """Inner Join: Includes only the time periods where data is available for all assets."""
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data_map:
            logger.warning("No data to align.")
            return pd.DataFrame()
            
        # data_map: {'AAPL': df1, 'TSLA': df2}
        renamed_dfs = []
        total_rows = sum(len(df) for df in data_map.values())
        logger.debug(f"Aligning {len(data_map)} assets. Total input rows: {total_rows}")
        
        for symbol, df in data_map.items():
            # Ensure index is datetime to prevent mismatch
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # Pull only necessary cols with symbol as a prefix
            temp = df.add_prefix(f"{symbol.upper()}_")
            renamed_dfs.append(temp)

        joint_df = pd.concat(renamed_dfs, axis=1, join='inner')
        
        before_drop = len(joint_df)
        joint_df.dropna(inplace=True)
        after_drop = len(joint_df)

        logger.info(f"Alignment Complete (Strict/Inner Join).")
        logger.info(f"  - Start Date: {joint_df.index.min()}")
        logger.info(f"  - End Date:   {joint_df.index.max()}")
        logger.info(f"  - Common Rows: {after_drop} (Dropped {before_drop - after_drop} rows of partial NaNs)")
        
        return joint_df

class FillAlignment(AlignmentStrategy):
    """Forward Fill: ถมข้อมูล (ดีกับ Lead/Lack แต่ต้องระวัง Noise)"""
    def align(self, data_dict):
        # Implementation of pd.concat with ffill()
        pass