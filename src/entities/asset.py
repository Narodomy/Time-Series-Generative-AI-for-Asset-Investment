import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_equity

logger = logging.getLogger(__name__)

class Asset:
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
    
    def to_returns(self, log=True, columns=['Close'], keep=False):
        """Convert Price to Return"""
        for col in columns:
            if col in self.data.columns:
                if log:
                    # Log Return: ln(Pt / Pt-1)
                    self.data[f"Log Returns"] = np.log(self.data[col]).diff()
                else:
                    # Simple Return: (Pt - Pt-1) / Pt-1
                    self.data[f"Returns"] = self.data[col].pct_change()
            else:
                logger.warning(f'Feature ({col}) is not found!')

        if not keep:
            self.data.drop(columns=columns, inplace=True)
            
        self.data.dropna(inplace=True)
        logger.debug(f'{self.symbol} converted to Returns (Log={log})')

    def inverse_returns(self, log=True, column=None, initialPrice=None)
    
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