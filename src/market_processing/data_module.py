import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .features import RollingFeatureEngineer

logger = logging.getLogger(__name__)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Slice data into windows [i : i + window]
        # Input (X): Sequence of length 'window_size'
        # Target (y): Typically the next step or a reconstruction for Path Gen tasks
        # Example: Input = indices 0 to 59, Target = indices 1 to 60 (Next-step prediction)
        window_data = self.data[idx : idx + self.window_size]
        target_data = self.data[idx + 1 : idx + self.window_size + 1]
        return window_data, target_data

# --- Main Class: The Manager ---
class MarketDataModule:
    def __init__(
        self,
        joint_df: pd.DataFrame,
        target_cols: list=None,
        window_size: int = 64,
        batch_size: int = 32,
        use_stats_features: bool =True,
        split_ratio: tuple = (0.7, 0.15, 0.15) # Train, Valid, Test
    ):
        self.df = joint_df
        self.stats_df = None
        self.window_size = window_size
        self.batch_size = batch_size
        self.use_stats_features = use_stats_features
        self.split_ratio = split_ratio

        # if not define the target_cols will take all of cols.
        self.target_cols = target_cols if target_cols else joint_df.columns.tolist()
        self.scaler = StandardScaler()

        # Placeholder for Datasets
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

        self._is_setup = False

    def setup(self):
        """Prepare data: Split -> Scale -> Window -> Tensor"""
        if self._is_setup:
            return
        
        logger.info("Setting up MarketDataModule...")
        
        logger.debug(f" Use states featyres: {self.use_stats_features}, DataFrame Index: {self.df.index}, Num of DataFrame: {len(self.df)}")
        if self.use_stats_features:
            engineer = RollingFeatureEngineer(window_size=self.window_size)
            stats_df = engineer.transform(self.df)
            self.stats_df = stats_df
            data = stats_df.values
            logger.debug(f" Num of Stats DataFrame: {len(stats_df)}")
        else:
            # 1. Time Series Split indices
            data = self.df[self.target_cols].values
            logger.debug(f" Num of Data: {len(data)}")
            
        n = len(data)
        
        train_end = int(n * self.split_ratio[0])
        validate_end = int(n * (self.split_ratio[0] + self.split_ratio[1]))

        train_data = data[:train_end]
        validate_data = data[train_end:validate_end]
        test_data = data[validate_end:]

        logger.debug(f" Split sizes - Train: {len(train_data)}, Val: {len(validate_data)}, Test: {len(test_data)}")

        # 2. Fit Scaler ONLY on Train data (Prevent Leakage)
        self.scaler.fit(train_data)

        # 3. Transform all sets
        train_scaled = self.scaler.transform(train_data)
        validate_scaled = self.scaler.transform(validate_data)
        test_scaled = self.scaler.transform(test_data)

        # 4. Create Sliding Windows & TensorDatasets
        self.train_dataset = self._create_tensor_dataset(train_scaled)
        self.validate_dataset = self._create_tensor_dataset(validate_scaled)
        self.test_dataset = self._create_tensor_dataset(test_scaled)

        self._is_setup = True
        logger.info(" MarketDataModule setup complete.")

    def _create_tensor_dataset(self, data_array):
        """Helper to create sliding windows"""
        X, y = [], []
        # Create sliding windows: 64-day input -> predict next step (or reconstruction)
        # Task-dependent: For Path Gen, Input and Target may be identical.
        for i in range(len(data_array) - self.window_size):
            X.append(data_array[i : i + self.window_size])
            y.append(data_array[i + 1 : i + self.window_size + 1]) # Next step prediction example
            
        return TensorDataset(
            torch.FloatTensor(np.array(X)), 
            torch.FloatTensor(np.array(y))
        )

    def _clean_and_inspect_features(self, df: pd.DataFrame, label: str = "Data") -> pd.DataFrame:
        """
        Generic Helper: 
        1. Cut Suffix Column name to shorter (Clean Name)
        2. Log to seek the Shape and sample with Label
        """
        if df is None or df.empty:
            logger.warning(f"[{label}] DataFrame is empty. Skipping inspection.")
            return df

        # 1. Clean Columns
        df_clean = df.copy()
        
        new_names = {}
        for col in df_clean.columns:
            clean_name = col.replace("_Close_Returns", "") \
                            .replace("_ret", "") \
                            .replace("_Close", "")
            new_names[col] = clean_name
            
        df_clean.rename(columns=new_names, inplace=True)

        # 2. Logging Inspection
        logger.info(f"====== Inspection: {label} ======")
        logger.info(f"  > Shape: {df_clean.shape}")
        
        # Format Column list
        col_str = ", ".join(df_clean.columns.tolist())
        logger.info(f"  > Columns: [{col_str}]")
        
        logger.debug(f"  > Head (First 3 rows):\n{df_clean.head(3).to_string()}")
        logger.info(f"===================================")

        return df_clean
    
    # --- DataLoader Accessors ---
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False) # If Shuffle Train is True means It's avaliable (Window is independent)
    
    def validate_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def inverse_transform(self, scaled_data: torch.Tensor or np.ndarray):
        """Transform data (Scaled) which be generated by a model. Let them back to the real world value (Returns)!"""
        is_tensor = False
        if torch.is_tensor(scaled_data):
            data_np = scaled_data.detach().cpu().numpy()
            is_tensor = True
        else:
            data_np = scaled_data

        # Handle shape: Scaler need the shape (N, Features)
        # Model output shape: (B, S, F) -> (Batch, Seq, Features)
        original_shape = data_np.shape
        if len(original_shape) == 3:
            # Flatten: (Batch * Seq, Features)
            flat_data = data_np.reshape(-1, original_shape[-1])
            inv_flat = self.scaler.inverse_transform(flat_data)
            inv_data = inv_flat.reshape(original_shape)
        else:
            inv_data = self.scaler.inverse_transform(data_np)
            
        return torch.FloatTensor(inv_data) if is_tensor else inv_data