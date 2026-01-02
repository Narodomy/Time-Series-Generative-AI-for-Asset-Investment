import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_equity

logger = logging.getLogger(__name__)

class SingleAsset:
    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data.copy() # OHLCV
        logger.debug(f"Initialized Asset: {self.symbol} with {len(data)} rows.")
    
    @classmethod
    def from_symbol(cls, symbol: str, freq="1d"):
        """Factory Method: New an Instance from download file (Optional)"""
        df = read_equity(symbol=symbol, freq=freq)
        if df is None or df.empty:
            raise ValueError(f"Cannot load data for {symbol}")
        return cls(symbol, df)

    @property
    def n_observed(self) -> int:
        return len(self.data)

    def __len__(self):
        return self.n_observed
    
    def to_returns(self, log=True, target_features=['Close']):
        """Convert Price to Return"""
        for col in target_features:
            if col in self.data.columns:
                if log:
                    # Log Return: ln(Pt / Pt-1)
                    self.data[f"{col}_ret"] = np.log(self.data[col]).diff()
                else:
                    # Simple Return: (Pt - Pt-1) / Pt-1
                    self.data[f"{col}_ret"] = self.data[col].pct_change()
            else:
                logger.warning(f'Feature ({col}) is not found!')
        self.data.dropna(inplace=True)
        logger.debug(f'{self.symbol} converted to Returns (Log={log})')
        
    def add_local_feature(self, func, **kwargs):
        """Put specific indicator such as RSI, MA"""
        self.data = func(self.data, **kwargs)

    """ 
    ==========================================
    [ADD] Volatility Calculation Methods
    ==========================================
    """
    
    def get_simple_vol(self, look_back=30, annualized=True, days_per_year=252):
        """Rolling Standard Deviation based on Close Price"""
        c = self.data['Close']
        # Log Return
        r = np.log(c.pct_change(1) + 1)
        vol = r.rolling(look_back).std()
        
        if annualized:
            vol = vol * np.sqrt(days_per_year) * 100
        return vol

    def get_ewma_vol(self, lamb=0.94, annualized=True, days_per_year=252):
        """Exponentially Weighted Moving Average Volatility"""
        c = self.data['Close']
        r = np.log(c.pct_change(1) + 1)
        
        # ใช้ Pandas EWM: alpha = 1 - lambda
        # adjust=False เพื่อให้คล้าย RiskMetrics Recursive formula
        vol = r.ewm(alpha=(1 - lamb), adjust=False).std()
        
        if annualized:
            vol = vol * np.sqrt(days_per_year) * 100
        return vol

    def get_parkinson_vol(self, look_back=30, annualized=True, days_per_year=252):
        """Parkinson Volatility (Uses High/Low) - Good for intraday range info"""
        if 'High' not in self.data.columns or 'Low' not in self.data.columns:
            print(f"Warning: {self.symbol} missing High/Low data for Parkinson Vol.")
            return pd.Series(dtype=float)

        h = self.data['High']
        l = self.data['Low']
        
        ln_sq = np.log(h / l) ** 2
        mean_ln_sq = ln_sq.rolling(look_back).mean()
        factor = 1.0 / (4 * np.log(2))
        vol = np.sqrt(factor * mean_ln_sq)

        if annualized:
            vol = vol * np.sqrt(days_per_year) * 100
        return vol

    def get_garman_klass_vol(self, look_back=30, annualized=True, days_per_year=252):
        """Garman-Klass Volatility (Uses OHLC) - More efficient estimator"""
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.data.columns for col in required):
            print(f"Warning: {self.symbol} missing OHLC data for Garman-Klass Vol.")
            return pd.Series(dtype=float)

        c = self.data['Close']
        o = self.data['Open']
        h = self.data['High']
        l = self.data['Low']

        ln_hl_sq = np.log(h / l) ** 2
        ln_oc_sq = np.log(o / c) ** 2
        
        # Garman-Klass Formula
        arg = 0.5 * ln_hl_sq - (2 * np.log(2) - 1) * ln_oc_sq
        vol = np.sqrt(arg.rolling(look_back).mean())

        if annualized:
            vol = vol * np.sqrt(days_per_year) * 100
        return vol
    
    def plot(self, column='Close'):
        """
        Simple plot for specific column (Price or Return)
        x: Time (Index)
        y: Value (Column)
        """
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found in {self.symbol}")
            return
            
        plt.figure(figsize=(10, 5))
        
        # X = Index (Time), Y = Data Column
        plt.plot(self.data.index, self.data[column], label=f"{self.symbol} {column}")
        
        plt.title(f"{self.symbol} - {column}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_volatility_comparison(self, look_back=30, lamb=0.94):
        """
        Plot comparison of all volatility estimators.
        """
        vol_simple = self.get_simple_vol(look_back=look_back)
        vol_ewma = self.get_ewma_vol(lamb=lamb)
        vol_park = self.get_parkinson_vol(look_back=look_back)
        vol_garman = self.get_garman_klass_vol(look_back=look_back)

        plt.figure(figsize=(15, 6))
        
        # Plotting
        if not vol_simple.empty:
            plt.plot(vol_simple.index, vol_simple, label=f"Simple ({look_back}d)", linewidth=1.5)
            
        if not vol_ewma.empty:
            plt.plot(vol_ewma.index, vol_ewma, label=f"EWMA ($\lambda={lamb:.2f}$)", linestyle='--')
            
        if not vol_park.empty:
            plt.plot(vol_park.index, vol_park, label=f"Parkinson ({look_back}d)", alpha=0.8)
            
        if not vol_garman.empty:
            plt.plot(vol_garman.index, vol_garman, label=f"Garman-Klass ({look_back}d)", alpha=0.8)

        plt.title(f"Volatility Estimates Comparison: {self.symbol}")
        plt.ylabel("Annualized Volatility (%)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()