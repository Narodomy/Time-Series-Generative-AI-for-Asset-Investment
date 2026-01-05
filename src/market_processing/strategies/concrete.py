import logging
import pandas as pd
from typing import Dict, List, Optional
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

class MarketFeatureAlignment(AlignmentStrategy):
    """Pull Only Returns, Volatility Columns"""
    def __init__(self, returns_patterns: List[str] = ['_Returns', '_ret'], volatility_patterns: List[str] = ['Vol_', 'GK_Vol'],):
        self.returns_patterns = returns_patterns
        self.volatility_patterns = volatility_patterns
        
        self.all_patterns = (
            self.returns_patterns +
            self.volatility_patterns
        )
        
    def align(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned_dfs = []
        
        for symbol, df in data_map.items():
            # 1. Filter Logic: Select a column if its name matches one of the patterns we've defined.
            # Use Logic: "If this word appears in the column name, keep it."
            selected_cols = [
                col for col in df.columns 
                if any(pat in col for pat in self.all_patterns)
            ]
            logger.debug(f"Symbol: {symbol}, DataFrame Column Names: {list(df.columns)}, Selected Columns: {selected_cols}")
            
            if not selected_cols:
                logger.warn(f"Warning There have no {selected_cols} columns in the data")
                continue
                
            temp = df[selected_cols].copy()
            
            # Handle Naming (Safety Check)
            # If column name has no a symbol as a prefix -> that must have! (for joining worst case)
            new_names = {}
            for col in temp.columns:
                # ถ้าชื่อ Column ยังไม่มี Symbol ให้เติมเข้าไปข้างหน้า
                if symbol not in col:
                    new_names[col] = f"{symbol}_{col}"
                    
            if new_names:
                temp.rename(columns=new_names, inplace=True)
                
            aligned_dfs.append(temp)
            
        if not aligned_dfs:
            raise ValueError("No valid features found based on provided patterns!")
                    
        joint_df = pd.concat(aligned_dfs, axis=1, join='inner')
        joint_df.dropna(inplace=True) # Drop NaN
        
        return joint_df