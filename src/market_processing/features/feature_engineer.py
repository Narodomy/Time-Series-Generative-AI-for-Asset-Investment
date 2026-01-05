import logging
import pandas as pd
import numpy as np
from typing import Tuple, Literal

logger = logging.getLogger(__name__)

class RollingFeatureEngineer:
    def __init__(self, window_size=64):
        self.window = window_size
        self.last_cov_matrix = None

    def transform(self, df_joint: pd.DataFrame) -> pd.DataFrame:
        """ 
        Transform Returns -> Rolling Stats (Vol + Corr) 
        Param df_joint must have:
            1. Returns (For an example APPL_Close_ret) -> to Correlation
            2. Calced Vol (For an example AAPL_Vol_GK) -> to Correlation
        """
        if df_joint.empty:
            raise ValueError("Input DataFrame is empty!")

        vol_cols = [c for c in df_joint.columns if 'Vol' in c]
        ret_cols = [c for c in df_joint.columns if 'Returns' in c or '_ret' in c]

        if not vol_cols or not ret_cols:
            raise ValueError(f"Columns missing! Found Vol: {len(vol_cols)}, Ret: {len(ret_cols)}")

        logger.debug(f"Processing Features: {len(vol_cols)} Vol cols, {len(ret_cols)} Return cols")

        # Calc Rolling Correlation (from Returns)
        # rolling().corr() will obtain MultiIndex (Date, Asset)
        df_ret = df_joint[ret_cols]
        rolling_corr = df_ret.rolling(window=self.window).corr()

        # Volatility
        # Cut first section (window-1) because Rolling Corr will be NaN
        df_vol = df_joint[vol_cols].iloc[self.window-1:]

        # Index Intersection
        valid_dates = df_vol.index.intersection(rolling_corr.index.get_level_values(0).unique())

        # Col for Correlation (Upper Triangle)
        # 'AAPL_Close_Returns' -> 'AAPL' to Corr
        asset_names = [c.replace('_Close_Returns', '').replace('_Returns', '').replace('_ret', '') for c in ret_cols]
        n_assets = len(asset_names)

        corr_feature_names = []
        rows, cols = np.triu_indices(n_assets, k=1) # k=1 means ingore diagonal
        for r, c in zip(rows, cols):
            # Name: Corr_AAPL_TSLA
            name = f"Corr_{asset_names[r]}_{asset_names[c]}"
            corr_feature_names.append(name)

        # Loop Flatten Matrix (Vectorization Step)
        features_list = []
        
        for date in valid_dates:
            try:
                # Pull volatility vector that day
                vol_vec = df_vol.loc[date].values
                
                # # Pull correlation matrix (N x N) that day
                # rolling_corr is a MultiIndex must use loc[date]
                corr_matrix = rolling_corr.loc[date].values
                
                # C. Flatten: Only Upper Triangle
                corr_flat = corr_matrix[rows, cols]
                
                # D. Combine: [Vol_1, Vol_2, ..., Corr_1_2, Corr_1_3, ...]
                feat_vec = np.concatenate([vol_vec, corr_flat])
                features_list.append(feat_vec)
                
            except KeyError as e:
                logger.warning(f"Data missing for date {date}: {e}")
                continue
        # Column = Vol + Corr
        final_cols = vol_cols + corr_feature_names
        
        result_df = pd.DataFrame(features_list, index=valid_dates, columns=final_cols)

        # Drop NaN
        if result_df.isnull().values.any():
            rows_before = len(result_df)
            result_df.dropna(inplace=True)
            logger.warning(f"Dropped {rows_before - len(result_df)} rows containing NaNs in final features.")

        return result_df