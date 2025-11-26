import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from typing import List, Optional
from sklearn.preprocessing import StandardScaler

class TSFinDataset(Dataset):
    def __init__(
        self, 
        series: pd.DataFrame, 
        features: Optional[List[str]] = None,
        window_size: int = 64,  # L: Length
        scaler: Optional[object] = None,
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
    
        # ถ้าเปิดใช้ Timestamp ก็ส่งเพิ่มไปใน Dict
        if self.use_timestamp and self.time_matrix is not None:
            time_sample = self.time_matrix[start:end]
            item["x_time"] = torch.tensor(time_sample, dtype=torch.float32) # [L, 4] (4 feature เวลา)
    
        return item

    def get_scaler(self):
        # Utilizing Inverse Transform for plot graph
        return self.scaler

    def get_n_features(self):
        return len(self.features)