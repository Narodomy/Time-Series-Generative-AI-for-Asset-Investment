import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_equity

logger = logging.getLogger(__name__)

log_ret_key = "(Log_Returns)"
simple_ret_key = "(Returns)"

class Asset:
    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data.copy() # OHLCV
        self.initial_prices: Dict[str, float] = {}
        self.return_type: Dict[str, str] = {}  # "log" | "simple"
        
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

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def device(self):
        return self.data.device

    def to(self, device: torch.device):
        """Moves the underlying tensor to the specified device."""
        self.data = self.data.to(device)
        return self

    def __len__(self):
        return self.n_observed
    
    def to_returns(self, log=True, columns=['Close'], keep=False):
        """ Convert Price to Returns. """
        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Feature ({col}) not found!")
                continue
    
            # store P0
            self.initial_prices[col] = float(self.data[col].iloc[0])
            self.return_type[col] = "Log" if log else "Simple"
    
            if log:
                # Log Return: ln(Pt / Pt-1)
                self.data[f"{col} {log_ret_key}"] = np.log(self.data[col]).diff()
            else:
                # Simple Return: (Pt - Pt-1) / Pt-1
                self.data[f"{col} {simple_ret_key}"] = self.data[col].pct_change()
    
        if not keep:
            self.data.drop(columns=columns, inplace=True)
            
        self.data.dropna(inplace=True)
        logger.debug(f'{self.symbol} converted to Returns (log={log})')

    def inverse_returns(self, columns=None, keep=False, initialPrices: dict | None = None):
        """ Reconstruct price path from returns. """
        price_source = initialPrices or self.initial_prices
        if not price_source:
            raise ValueError("Initial prices (P0) not available")

        if columns is None:
            columns = [
                c for c in self.data.columns
                if c.endswith(log_ret_key) or c.endswith(simple_ret_key)
            ]
        
        if not columns:
            raise ValueError("No return columns found")

        for col in columns:
            feature, ret_type = col.rsplit(" ", 1)
    
            if feature not in price_source:
                raise ValueError(f"No initial price for '{feature}'")
    
            r = self.data[col].values
            P0 = price_source[feature]
    
            if ret_type == log_ret_key:
                price = P0 * np.exp(np.cumsum(r))
            elif ret_type == simple_ret_key:
                price = P0 * np.cumprod(1.0 + r)
            else:
                raise ValueError(f"Unknown return type: {ret_type}")
    
            out_col = feature
            self.data[out_col] = price
            
        if not keep:
            self.data.drop(columns=columns, inplace=True)
            
        return self
    
    def add_local_feature(self, func, **kwargs):
        """Put specific indicator such as RSI, MA"""
        self.data = func(self.data, **kwargs)
    
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