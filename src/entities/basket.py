import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Optional
from .asset import Asset
from strategies import AlignmentStrategy

logger = logging.getLogger(__name__)

class Basket:
    def __init__(self, symbols: List[str], device: torch.device= torch.device("cuda")):
        self.symbols = symbols
        self.assets: Dict[str, Asset] = {}
        self.stats = None

        self._device = device
        # Load data logic here...
        logger.debug(f"Initialized Asset Basket: {self.symbols} with {len(self.assets)} assets which loaded.")
        
    @property
    def n_planned(self) -> int:
        return len(self.symbols)

    @property
    def n_loaded(self) -> int:
        return len(self.assets)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def device(self):
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Asset: {self.symbol} is using {self._device} device.")
        return self._device
    
    def __len__(self):
        return self.n_loaded
        
    def add_symbol(self, symbol: str):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.debug(f"Added symbol: {symbol}")
    
    def add_asset(self, asset: Asset):
        if asset.data is None or asset.data.empty:
            logger.warning(f"Attempted to add empty asset: {asset.symbol}. Skipping.")
            return
        if asset.symbol not in self.symbols:
            self.symbols.append(asset.symbol)
    
        self.assets[asset.symbol] = asset

    def load_asset(self, symbol: str, freq="1d") -> bool:
        try:
            logger.debug(f"Attempting to load {symbol}...")
            asset = Asset.from_symbol(symbol, freq=freq)
            self.assets[symbol] = asset
            logger.info(f"Successfully loaded {symbol} ({len(asset)} rows).")
            return True
        except FileNotFoundError:
            logger.error(f"File not found for symbol: {symbol}")
            return False
        except ValueError as ve:
            logger.warning(f"Data validation failed for {symbol}: {ve}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading {symbol}: {e}", exc_info=True)
            return False
            
    def load_all_assets(self, freq="1d"):
        logger.info(f"Starting batch load for {len(self.symbols)} symbols...")
        success_count = 0
        
        for symbol in self.symbols:
            if self.load_asset(symbol, freq):
                success_count += 1
        
        logger.info(f"Batch load complete. Success: {success_count}/{self.n_planned}. Total assets in basket: {self.n_loaded}")

    def align(self, strategy: AlignmentStrategy, inplace: bool = True) -> pd.DataFrame:
        """
        Aligns all assets based on the provided strategy.
        
        Args:
            strategy: The logic to align dates (e.g., Inner Join, Outer Join).
            inplace: If True, updates the internal .data of each Asset to match the aligned index.
                     (Required True if you want to call .to_tensor() afterwards)
        Returns:
            pd.DataFrame: A single DataFrame containing aligned data (MultiIndex or wide format).
        """
        if not self.assets:
            logger.warning("Basket is empty! Cannot perform alignment.")
            return pd.DataFrame()

        # Gather raw data
        data_map = {t: a.data for t, a in self.assets.items()}

        try:
            # Perform Alignment (Strategy Pattern)
            aligned_df = strategy.align(data_map)
            logger.debug(f"Aligned data shape: {aligned_df.shape}")

            # If inplace, update individual assets to match the aligned index
            if inplace:
                common_index = aligned_df.index
                for symbol, asset in self.assets.items():
                    asset.data = asset.data.loc[asset.data.index.intersection(common_index)]
            
                logger.info(f"Assets updated in-place to aligned index (Length: {len(common_index)})")
            
            return aligned_df
            
        except Exception as e:
            logger.error(f"Alignment failed: {e}", exc_info=True)
            raise e

    def get_stats_summary(self, column='Close_Returns') -> pd.DataFrame:
        """
        Calculate descriptive statistics for all assets.
        Returns a DataFrame containing Mean, Std, Min, Max, Skew, Kurtosis, etc.
        """
        if not self.assets:
            logger.warning("Basket is empty. No stats to calculate.")
            return pd.DataFrame()

        stats_list = []
        
        for symbol, asset in self.assets.items():
            if column not in asset.data.columns:
                continue
                
            series = asset.data[column].dropna() # Cut NaN first
            
            stats = {
                'symbol': symbol,
                'count': len(series),
                'mean': series.mean(),
                'std': series.std(),      # Volatility (Daily)
                'min': series.min(),
                '25%': series.quantile(0.25),
                '50%': series.median(),
                '75%': series.quantile(0.75),
                'max': series.max(),
                'skew': series.skew(),    # Skewness (Tell Tail Risk)
                'kurt': series.kurtosis() # Prominence (Tell Extreme Events)
            }
            stats_list.append(stats)
            
        df_stats = pd.DataFrame(stats_list)
        
        if not stats.empty:
            df_stats.set_index('symbol', inplace=True)
            
            # 1. Avg of All Stats (such as Avg Return, Avg Volatility of Basket)
            total_row = df_stats.mean()
            
            # 2. But 'count' must (Sum) not avg.
            total_row['count'] = df_stats['count'].sum()
            
            # 3. Add latest row with 'TOTAL_AVG' name
            df_stats.loc['TOTAL_AVG'] = total_row
            
        return df_stats

    def to_tensor(self, features: list[str], device: torch.device = None) -> np.ndarray:
        """
        Stack tensors from all loaded assets.
        Shape: [L, N, F] (Length, Num_Assets, Features)
        """
        if not self.assets:
            raise ValueError("Basket is empty!")

        # Collect tensors from each asset
        tensor_list = []
        for symbol in self.symbols:
            if symbol in self.assets:
                # Call Asset method
                asset_tensor = self.assets[symbol].to_tensor(features, device)
                tensor_list.append(asset_tensor)
        
        # Stack along dimension 1 (Dimension N)
        # Asset: [L, F] -> Stack dim=1 -> [L, N, F]
        basket_tensor = torch.stack(tensor_list, dim=1)
        
        return basket_tensor
    
    def plot_assets(self, column='Close_Returns', title="Basket Composition"):
        """
        Plot all assets in the basket on the same chart.
        """
        if not self.assets:
            logger.warning("Basket is empty. Nothing to plot.")
            return

        plt.figure(figsize=(12, 6))

        plotted_count = 0
        for symbol, asset in self.assets.items():
            if column in asset.data.columns:
                plt.plot(asset.data.index, asset.data[column], label=symbol, alpha=0.7)
                plotted_count += 1
            else:
                logger.warning(f"Asset {symbol} does not have column '{column}'")

        if plotted_count > 0:
            plt.title(f"{title} ({column})")
            plt.xlabel("Date")
            plt.ylabel(column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            logger.warning(f"No assets found with column '{column}'. Did you calculate returns?")