import logging
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from typing import List, Dict, Optional
from .asset import Asset
from strategies import AlignmentStrategy

logger = logging.getLogger(__name__)

class Basket:
    def __init__(self, symbols: List[str], device: torch.device= torch.device("cuda")):
        self.symbols = symbols
        self.assets: Dict[str, Asset] = {}
        
        self._data: Optinal[pd.DataFrame] = None
        self._device = device
        # Load data logic here...
        logger.debug(f"Initialized Asset Basket: {self.symbols} with {len(self.assets)} assets which loaded.")


    @property 
    def data(self) -> pd.DataFrame:
        # if self._data is not None:
        #     return self._data
            
        if self.assets:
            return pd.concat(
                {s: a.data for s, a in self.assets.items()}, 
                axis=1, 
                keys=self.assets.keys() # MultiIndex (Symbol, Feature)
            )

        return pd.DataFrame()
        
    @property
    def device(self):
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Asset: {self.symbol} is using {self._device} device.")
        return self._device

    def get_unique_features(self) -> List[str]:
        if not self.assets:
            return []
            
        unique_cols = {col for asset in self.assets.values() for col in asset.data.columns}        
        return sorted(list(unique_cols))

    def to_returns(self, features: list, log: bool, keep: bool=False):
        for symbol, asset in self.assets.items():
            asset.to_returns(log=log, columns=features, keep=keep)
    
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
        
        logger.info(f"Batch load complete. Success: {success_count}/{len(self.symbols)}. Total assets in basket: {len(self.assets)}")

    def align(self, strategy: AlignmentStrategy, inplace: bool = True) -> pd.DataFrame:
        if not self.assets:
            logger.warning("Basket is empty! Cannot perform alignment.")
            return pd.DataFrame()

        # Gather raw data
        data_map = {t: a.data for t, a in self.assets.items()}

        try:
            # Perform Alignment (Strategy Pattern)
            aligned_df = strategy.align(data_map)
            self._data = aligned_df
            
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
        Shape: [T, A, F] (Time, Assets, Features)
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
        # Asset: [T, F] -> Stack dim=1 -> [T, A, F]
        basket_tensor = torch.stack(tensor_list, dim=1)
        
        return basket_tensor

    def get_feature(self, feature: str, drop_empty: bool = True) -> pd.DataFrame:
        if self.data.empty:
            logger.warning("Basket data is empty.")
            return pd.DataFrame()
        try:
            # Slice specific feature from MultiIndex columns
            df_slice = self.data.xs(key=feature, level=1, axis=1)
            
            if drop_empty:
                # Drop columns (assets) that are entirely NaN
                df_slice = df_slice.dropna(axis=1, how='all')

            return df_slice
        except KeyError:
            logger.warning(f"Feature '{feature}' not found in any asset.")
            return pd.DataFrame()

            
    def plot_assets(self, column, title="Basket Composition"):
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

    def plot_cumulative_returns(self, return_col, title="Cumulative Performance"):
        if self.data.empty: return

        try:
            df_returns = self.get_feature(return_col, drop_empty=True)
            if df_returns.empty: return

            # Calculate Cumulative Return: (1+r).cumprod() - 1
            df_cum = (1 + df_returns.fillna(0)).cumprod() - 1

            plt.figure(figsize=(12, 6))
            for symbol in df_cum.columns:
                plt.plot(df_cum.index, df_cum[symbol], label=symbol, linewidth=1.5)

            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.title(f"{title} (based on {return_col})")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Format Y-axis as percentage
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            plt.show()

        except Exception as e:
            logger.error(f"Plotting cumulative returns failed: {e}")

    def plot_distribution(self, column: str= "Close", kde: bool= True):
        if self.data.empty:
            logger.warning("Basket is empty.")
            return

        try:
            df_slice = self.get_feature(column, drop_empty=True)
            if df_slice.empty: 
                logger.warning(f"No data found for feature '{column}'")
                return

            plt.figure(figsize=(12, 6))
            for symbol in df_slice.columns:
                sns.histplot(
                    df_slice[symbol], 
                    kde=kde, 
                    label=symbol, 
                    element="step", 
                    stat="density", 
                    alpha=0.3
                )
            
            plt.title(f"Return Distribution ({column})")
            plt.xlabel("Return")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            
        except Exception as e:
            logger.error(f"Plotting distribution failed: {e}")