import os
import torch
import numpy as np
import pandas as pd

from .csv_reader import read_equity
from utils.paths import PRICE_DIR
from utils import viz_single_timeline, viz_single_window, viz_group_timeline, viz_group_window
from tqdm import tqdm
from datetime import date
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class PortfolioDataset(Dataset):
    def __init__(
        self,
        security_basket_dataset: "SecurityBasketDataset",
        assets_dict: Dict[str, pd.DataFrame], # Raw Data { "AAPL": df, "TSLA": df, ...}
        features: Optional[list] = None,
        window_size: int = 64,
        use_timestamp: bool = False,
        use_log_return: bool = False,
        log_return_features: Optional[list] = None,
        time_col: str = 'Date',
        viz_config: dict = None
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
        for symbol, df in assets_dict.items():
            ts_ds = TimeSeriesDataset(
                series=df,
                features=features,
                window_size=window_size,
                use_timestamp=use_timestamp,
                time_col=time_col,
                scaler = None,
                use_log_return=use_log_return,
                log_return_features=log_return_features
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
        return len(self.security_basket_dataset)

    def __getitem__(self, idx):
        """ 1. We must know that what date and symbol are in the SecurityBasketDataset """
        security_basket = self.security_basket_dataset[idx]
        current_date = security_basket["date"] # Start Date
        target_symbols = security_basket["symbols"]

        batch_x = []
        batch_x_time = []
        valid_symbols = []
        
        """ 2. Loop each security in the list """
        for symbol in target_symbols:
            if symbol not in self.asset_datasets:
                continue
            dataset = self.asset_datasets[symbol]
            date_map = self.asset_date_maps[symbol]

            # Check current date that what security has ?
            if current_date in date_map: 
                row_idx_start = date_map[current_date]
                start_idx = row_idx_start
                
                if start_idx < len(dataset):
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

    def scan(self):
        """ Scan: Check all the timeline that what date, ready or skipped? 
            Return: Daily Status DataFrame
        """
        print(f"Scanning dataset ({len(self)} time steps)...")
        records = []
            
        for i in tqdm(range(len(self.security_basket_dataset)), desc="Scanning"):
            basket_data = self.security_basket_dataset[i]
            current_date = basket_data['date']
            target_symbols = basket_data['symbols']
            
            valid_count = 0
            for symbol in target_symbols:
                if symbol not in self.asset_datasets:
                    continue
                
                date_map = self.asset_date_maps[symbol]
                if current_date in date_map:
                    row_idx = date_map[current_date]
                    ts_dataset = self.asset_datasets[symbol]
                    total_len = len(ts_dataset.dates)
                    if row_idx + self.window_size <= total_len:
                        valid_count += 1
            
            status = 'Ready' if valid_count > 0 else 'Skipped'
            records.append({
                'Index': i,
                'Date': current_date,
                'Universe': len(target_symbols),
                'Assets': valid_count,
                'Status': status
            })
            
        df = pd.DataFrame(records)
        
        ready_count = len(df[df['Status'] == 'Ready'])
        print(f"\n--- Scan Complete ---")
        print(f"Total Days: {len(df)}")
        print(f"Ready:      {ready_count} days")
        print(f"Skipped:    {len(df) - ready_count} days (Holiday/No Data)")
        
        return df
        
    def preview(self, idx):
        print(f"\n--- Preview Index: {idx} ---")
        basket_data = self.security_basket_dataset[idx]
        print(f"Date: {basket_data['date'].date()}")
        print(f"Universe: {len(basket_data['symbols'])} symbols")
        item = self.__getitem__(idx)
        
        if item is None:
            print("Status: [SKIPPED] (Holiday or insufficient history)")
        else:
            print(f"Status: [READY]")
            print(f"  - Input Shape: {item['x'].shape} (Assets, Window, Features)")
            if 'symbols' in item:
                print(f"  - Sample Symbols: {item['symbols'][:3]} ...")

    def plot_all_assets_timeline(self, start_index: int = 0, feature: str = "Close"):
        """Watch Asset Overview start date at this index """
        viz_group_timeline(self, start_index, feature)

    def plot_all_assets_window(self, index: int, feature: str = "Close"):
        """Watch Asset Specific the Window at this index """
        viz_group_window(self, index, feature)
        
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
        use_log_return: bool = False,
        log_return_features: Optional[List[str]] = None,
        time_col: Optional[str] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.use_timestamp = use_timestamp
        self.use_log_return = use_log_return
        self.log_return_features = log_return_features

        # 1. Prepare Data
        data = series.copy()
        
        if self.use_log_return:
            targets = log_return_features if log_return_features else ['Close']
            for col in targets:
                if col in data.columns:
                    # คำนวณ Log Return: ln(Pt) - ln(Pt-1)
                    data[col] = np.log(data[col]).diff()
            
            # *** สำคัญ: ใช้ fillna(0) แทน dropna() ***
            # เพื่อให้จำนวนบรรทัดเท่าเดิม! Index ที่ PortfolioDataset ถืออยู่จะได้ไม่พัง
            data = data.fillna(0)
            
        """ 1. Care the target features! """
        if features is None:
            self.features = [c for c in series.columns if c != time_col]
        else:
            self.features = features
            
        """ 1.1 filter only target features! """
        self.raw_values = data[self.features].values.astype(np.float32)
      
        values = data[self.features].values

        """ 1.2 Keep date time """
        if time_col and time_col in data.columns:
            self.dates = data[time_col].values
        else:
            self.dates = np.arange(len(data))
    
        """ 2. Scaler """
        self.scaler = scaler if scaler is not None else StandardScaler()
        if hasattr(self.scaler, 'mean_'):
            self.is_fitted = True
        else:
            self.is_fitted = False        
        
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

    def plot_timeline(self, start_idx: int = 0, feature: str = "Close"):
        """Plot all data begin at start_idx util the end"""
        viz_single_timeline(self, start_idx, feature)

    def plot_window(self, start_idx: int, feature: str = "Close"):
        """Plot specific Window begin at start_idx"""
        viz_single_window(self, start_idx, feature)





