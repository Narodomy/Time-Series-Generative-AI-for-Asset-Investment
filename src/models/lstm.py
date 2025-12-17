import torch
import torch.nn as nn

from .modules import SinusoidalPositionEmbeddings

class DiffusionLSTM(nn.Module):
    def __init__(self, features_in, hidden_dim=64, num_layers=2):
        super().__init__()
        self.name = "DiffusionLSTM"
        
        # 1. Time Embedding (ใช้ตัวเดิมจากโพสต์ก่อนหน้า)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. Input Projector
        self.input_proj = nn.Linear(features_in, hidden_dim)
        
        # 3. LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True # ให้มองเห็นทั้งอดีตและอนาคต (สำคัญมาก!)
        )
        
        # 4. Output Projector (เนื่องจากเป็น Bi-LSTM output จะเป็น hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim * 2, features_in)

    def forward(self, x, t):
        # x: [B, L, C]
        # t: [B]
        
        # Embed Time
        t_emb = self.time_mlp(t) # [B, hidden_dim]
        
        # Project Input
        x_emb = self.input_proj(x) # [B, L, hidden_dim]
        
        # Add Time Embedding (บวกเข้าไปทุกๆ time step)
        x_emb = x_emb + t_emb[:, None, :]
        
        # LSTM Processing
        # output shape: [B, L, hidden_dim * 2]
        lstm_out, _ = self.lstm(x_emb) 
        
        # Final Prediction
        return self.output_proj(lstm_out)