import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MarketMaskedDataset(Dataset):
    def __init__(self, 
                 basket_df: pd.DataFrame, 
                 window_size: int = 60, 
                 mask_size: int = 2,
                 features_pipeline = None):
        
        self.data = basket_df
        self.window_size = window_size
        self.mask_size = mask_size # จำนวน step ที่จะปิดบัง (Future)
        self.features = features_pipeline
        
    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # 1. Slice Window
        # Shape: (Window_Size, Num_Assets)
        window_data = self.data.iloc[idx : idx + self.window_size].values
        
        # 2. Apply Dynamic Features (Optional)
        # ตรงนี้ถ้าอยากคำนวณ Correlation ของ Window นี้ส่งเข้าไปด้วย
        # corr_feat = self.features.calc_corr(window_data)
        
        # 3. Create Masking (Task Setup)
        # สมมติ Mask 2 ตัวท้าย (Future) ให้เป็น 0 หรือ Token พิเศษ
        src = window_data.copy()
        src[-self.mask_size:, :] = 0 # Masking the future (Return = 0)
        
        # 4. Target (Ground Truth)
        tgt = window_data # ของจริงมีครบ
        
        # 5. Mask Indicator (บอก Model ว่าตรงไหนโดน Mask)
        mask_indicator = np.zeros_like(src)
        mask_indicator[-self.mask_size:, :] = 1
        
        return {
            'input': torch.FloatTensor(src),
            'target': torch.FloatTensor(tgt),
            'mask': torch.FloatTensor(mask_indicator),
            # 'global_feat': torch.FloatTensor(corr_feat) # ถ้ามี
        }