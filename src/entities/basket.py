import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from .asset import SingleAsset
from market_processing.strategies.base import AlignmentStrategy

logger = logging.getLogger(__name__)

class AssetBasket:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.assets: Dict[str, SingleAsset] = {}
        # Load data logic here...
        logger.debug(f"Initialized Asset Basket: {self.symbols} with {len(self.assets)} assets which loaded.")
        
    @property
    def n_planned(self) -> int:
        return len(self.symbols)

    @property
    def n_loaded(self) -> int:
        return len(self.assets)

    def __len__(self):
        return self.n_loaded
        
    def add_symbol(self, symbol: str):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.debug(f"Added symbol: {symbol}")
    
    def add_asset(self, asset: SingleAsset):
        if asset.data is None or asset.data.empty:
            logger.warning(f"Attempted to add empty asset: {asset.symbol}. Skipping.")
            return
        if asset.symbol not in self.symbols:
            self.symbols.append(asset.symbol)
    
        self.assets[asset.symbol] = asset

    def load_asset(self, symbol: str, freq="1d") -> bool:
        try:
            logger.debug(f"Attempting to load {symbol}...")
            asset = SingleAsset.from_symbol(symbol, freq=freq)
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

    def get_joint_data(self, alignment_strategy: AlignmentStrategy) -> pd.DataFrame:
        if not self.assets:
            logger.warning("Basket is empty! Cannot perform alignment.")
            return pd.DataFrame()

        data_map = {t: a.data for t, a in self.assets.items()}
        
        try:
            aligned_df = alignment_strategy.align(data_map)
            logger.debug(f"Aligned data shape: {aligned_df.shape}")
            return aligned_df
        except Exception as e:
            logger.error(f"Alignment strategy failed: {e}", exc_info=True)
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
        
        if not df_stats.empty:
            df_stats.set_index('symbol', inplace=True)
            
            # 1. Avg of All Stats (such as Avg Return, Avg Volatility of Basket)
            total_row = df_stats.mean()
            
            # 2. But 'count' must (Sum) not avg.
            total_row['count'] = df_stats['count'].sum()
            
            # 3. Add latest row with 'TOTAL_AVG' name
            df_stats.loc['TOTAL_AVG'] = total_row
            
        return df_stats
    
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
            plt.legend() # โชว์ชื่อหุ้นตรงมุม
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            logger.warning(f"No assets found with column '{column}'. Did you calculate returns?")