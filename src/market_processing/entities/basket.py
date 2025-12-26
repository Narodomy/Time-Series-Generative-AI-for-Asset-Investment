import logging
import pandas as pd
from typing import List
from .asset import SingleAsset
from utils import read_equity
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
        self.symbols = list(dict.fromkeys(self.symbols + [symbol]))
    
    def add_asset(self, asset: SingleAsset):
        self.assets[asset.symbol] = asset

    def load_asset(self, symbol: str, freq="1d"):
        self.assets[symbol] = SingleAsset(symbol, read_equity(symbol=symbol, freq=freq))
        logger.debug(f"Loaded {symbol}, freq={freq} with ({len(self.assets[symbol])}) rows in the asset.")
    
    def load_all_assets(self, freq="1d"):
        for symbol in self.symbols:
            self.load_asset(symbol, freq)
        logger.debug(f"Loaded all {len(self.symbols)} symbols at freq={freq} with ({len(self.assets)}) assets in the asset basket.")

    def get_joint_data(self, alignment_strategy: AlignmentStrategy) -> pd.DataFrame:
        """
        พระเอกอยู่ตรงนี้: ส่ง Strategy เข้ามาเพื่อบอกว่าจะรวมร่างยังไง
        return: DataFrame ใหญ่ที่มี Column เป็น MultiIndex (symbol, Feature)
        """
        data_map = {t: a.data for t, a in self.assets.items()}
        return alignment_strategy.align(data_map)