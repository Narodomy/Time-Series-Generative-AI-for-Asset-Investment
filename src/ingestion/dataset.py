import os
import torch
import numpy as np
import pandas as pd

from .csv_reader import read_equity
from utils.paths import PRICE_DIR
from tqdm import tqdm
from datetime import date
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class PortfolioDataset(Dataset):
    def __init__(
        self,
        security_basket_dataset: "SecurityBasketDataset",
        asset_dict: Dict[str, pd.DataFrame], # Raw Data { "AAPL": df, "TSLA": df, ...}
        features: Optional[list] = None,
        window_size: int = 64,
        use_timestamp: bool = False,
        time_col: str = 'Date'
    ):
        self.security_basket_dataset = security_basket_dataset
        self.window_size = window_size
        self.use_timestamp = use_timestamp

        """ 1. Create a new 'Asset Group':
            They're converted to Time SeriesDataset 
        """
        self.asset_datasets = {}
        self.asset_date_maps = {}

        print(f"Initializing Portfolio with {len(assets_dict)} assets...")
        for symbol, df in asset_dict.items():
            ts_ds = TimeSeriesDataset(
                series=df,
                features=features,
                window_size=window_size,
                use_timestamp=use_timestamp,
                time_col=time_col,
                scaler = StandardScaler()
            )
            self.asset_datasets[symbol] = ts_ds

            """ Create a new Date Map -> Security Basket is most important! to search 0, 1
                We must know the position of SecurityBasketDataset says '2023-01-05'
                It will be matched what number of row in that asset's df 
            """
            if time_col in df.columns:
                dates = pd.to_datetime(df[time_col])
                # Date Map -> Index Row in original Dataframe
                self.asset_date_maps[symbol] = { date: i for i, date in enumerate(dates) }

    def __len__(self):
        return self.len(self.security_basket_dataset)

    def __getitem__(self, idx):
        """ 1. We must know that what date and symbol are in the SecurityBasketDataset """
        security_basket = self.security_basket_dataset[idx]
        current_date = security_basket["date"]
        target_symbols = security_basket["symbols"]

        batch_x = []
        batch_x_time = []
        valid_symbols = [] # Collect only actual Symbol bacause sometime Index has the name but data be lacked 

        """ 2. Loop each security in the list """
        for symbol in target_symbols:
            if symbol not in self.asset_datasets:
                continue
            dataset = self.asset_datasets[symbol]
            date_map = self.asset_date_maps[symbol]

            # Check current date that what security has ?
            if current_date in date_map: 
                row_idx_at_date = date_map[current_date]

                """ Logic to Slice the window 
                    TimeSeriesDataset[i] will pull the time range at [i : i + window_size]
                    if we want to end data at row_idx_at_date (include that day)
                    so i + window_size = row_idx_at_date + 1 (because slice not include the last one)
                    i = row_idx_at_date + 1 - window_size
                """
                start_idx = row_idx_at_date - self.window_size + 1
                
                # if start_idx < 0 means backward data is not enough 
                if start_idx >= 0: 
                    sample = dataset[start_idx]
                    batch_x.append(sample["x"])
                    if self.use_timestamp and "x_time" in sample:
                        batch_x_time.append(sample["x_time"])
                        
                    valid_symbols.append(symbol)
        """ 3. Stack all data in the one tensor """
        if len(batch_x) > 0:
            out_x = torch.stack(batch_x) # Shape: [N assets, Window size, N Features]
            result = {
                "date": str(current_date.date()),
                "x": out_x,
                "symbols": valid_symbols,
            }

            if len(batch_x_time) > 0:
                result["x_time"] = torch.stack(batch_x_time)
            
            return result
        else:
            return None # In case holiday or no data in that day!
            
class SecurityBasketDataset(Dataset):
    def __init__(self, csv_file, start_date, end_date=date.today(), freq="D"):
        self.n_assets = None
        
        self.df = pd.read_csv(csv_file)
        self.df["Date added"] = pd.to_datetime(self.df['Date added'], errors='coerce')
        self.df["Date added"] = self.df["Date added"].fillna(pd.Timestamp("1900-01-01"))

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.timeline = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        self.all_symbols = self.df['Symbol'].unique()
        
    def __len__(self):
        return len(self.timeline)

    def __getitem__(self, idx):
        current_date = self.timeline[idx]

        mask = self.df["Date added"] <= current_date
        active_companies = self.df[mask].copy()
        active_companies = active_companies.sort_values(by=["Date added", "Symbol"])

        if self.n_assets is not None:
            active_companies = active_companies.head(self.n_assets)
        
        symbols = active_companies["Symbol"].values
        sample = {
            "date": current_date,
            "symbols": symbols,
            "num_assets": len(symbols),
            "features": self.df.columns,
        }
        
        return sample

    def load_assets(self, data_dir=PRICE_DIR, interval="1d", time_col="Date", extension=".csv"):
        """ Load CSV File
            Args:
                data_dir (str): Directory path
                time_col (str): Time Column name
                extension (str): .file
            Returns:
                dict: { 'SYMBOL': dataframe, ... }
        """
        print(f"Loading assets for {len(self.all_symbols)} potential symbols...")
        
        assets_dict = {}
        missing_count = 0
        
        for symbol in tqdm(self.all_symbols, desc="Reading CSVs"):
            try:
                df = read_equity(symbol, interval=interval)
                df = df.reset_index()
                assets_dict[symbol] = df
            except Exception as e:
                missing_count += 1

        print(f"Done! Loaded {len(assets_dict)} assets. (Missing/Error: {missing_count})")
        return assets_dict

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        series: pd.DataFrame, 
        features: Optional[List[str]] = None,
        scaler: Optional[object] = None,
        window_size: int = 64,  # Length of Window (L channel)
        use_timestamp: bool = False,
        time_col: Optional[str] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.use_timestamp = use_timestamp
        
        """ 1. Care the target features! """
        if features is None:
            self.features = [c for c in series.columns if c != time_col]
        else:
            self.features = features
        """ 1.1 filter only target features! """
        self.raw_values = series[self.features].values.astype(np.float32)
      
        values = series[self.features].values

        """ 2. Scaler """
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = scaler is not None
        if not self.is_fitted:
            self.scaled_values = self.scaler.fit_transform(self.raw_values)
            self.is_fitted = True
        else:
            self.scaled_values = self.scaler.transform(self.raw_values)

        """ 3. Time Features """
        self.time_matrix = None
        if self.use_timestamp and time_col:
            """ Convert to datetime object for make sure """
            dt_series = pd.to_datetime(series[time_col])
            
            """ 
                Pull the significant features
                Normalize in range [-0.5, 0.5] or [0, 1] to Model learn easily
                such as: Hour of Day, Day of Week, Month of Year
            """
            time_feats = np.stack([
                dt_series.dt.hour.values / 23.0,      # 0-23 -> 0-1
                dt_series.dt.dayofweek.values / 6.0,  # 0-6  -> 0-1
                dt_series.dt.day.values / 31.0,       # 1-31 -> ~0-1
                dt_series.dt.month.values / 12.0      # 1-12 -> ~0-1
            ], axis=1).astype(np.float32)
            
            self.time_matrix = time_feats # Shape: [Total_Len, 4]
            
        self.n_samples = len(self.scaled_values) - self.window_size + 1
        
    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.window_size
    
        sample = self.scaled_values[start:end]
        sample_raw = self.raw_values[start:end]
    
        item = {
            "x": torch.tensor(sample, dtype=torch.float32),      # [L, C]
            "x_raw": torch.tensor(sample_raw, dtype=torch.float32), 
        }
    
        # In case Timestamp = True will be sent to add Dict
        if self.use_timestamp and self.time_matrix is not None:
            time_sample = self.time_matrix[start:end]
            item["x_time"] = torch.tensor(time_sample, dtype=torch.float32) # [L, 4] (4 feature เวลา)
    
        return item

    def get_scaler(self):
        # Utilizing Inverse Transform for plot graph
        return self.scaler

    def get_n_features(self):
        return len(self.features)