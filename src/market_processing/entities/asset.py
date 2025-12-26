import logging
import pandas as pd

logger = logging.getLogger(__name__)

class SingleAsset:
    def __init__(self, ticker: str, data: pd.DataFrame):
        self.ticker = ticker
        self.data = data # OHLCV
        logger.debug(f"Initialized Asset: {self.ticker} with {len(data)} rows.")
        
    def to_returns(self, log=True, target_features=['Close']):
        """Convert Price to Return"""
        self.data
        for col in target_features:
            if col in data.columns:
                self.data[col] = np.log(self.data[col]).diff()
            else:
                logger.warning(f'Feature ({col}) is not found!')  
        logger.debug(f'{self.ticker} is conveted to Returns!')
    
    def add_local_feature(self, func, **kwargs):
        """Put specific indicator such as RSI, MA"""
        self.data = func(self.data, **kwargs)