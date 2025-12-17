import torch
import torch.nn as nn

from .modules import SinusoidalPositionEmbeddings

class DiffusionTransformer(nn.Module):
    def __init__(self, features_in, d_model=64, nhead=4, num_layers=4, dropout=0.1, max_len=500):
        super().__init__()
        self.name = "DiffusionTransformer"
        self.d_model = d_model
        
        # 1. Input Projection: แปลง Feature เดิม (C) ให้เป็นขนาด Model (d_model)
        self.input_proj = nn.Linear(features_in, d_model)
        
        # 2. Time Embedding (สำหรับ t): สำคัญมากสำหรับ DDPM
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3. Positional Encoding (สำหรับ Sequence): ให้รู้ว่าจุดไหนมาก่อนมาหลัง
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # 4. Transformer Encoder (หัวใจหลัก แทน U-Net/LSTM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Projection: แปลงกลับเป็นขนาด Feature เท่าเดิมเพื่อทำนาย Noise
        self.output_proj = nn.Linear(d_model, features_in)

    def forward(self, x, t):
        """
        x shape: [Batch, Length, Channels]
        t shape: [Batch]
        """
        B, L, C = x.shape
        
        # --- 1. Process Input ---
        x_emb = self.input_proj(x)  # [B, L, d_model]
        
        # + Positional Encoding (Sequence Position)
        x_emb = x_emb + self.pos_encoder[:, :L, :]
        
        # --- 2. Process Time Step (t) ---
        t_emb = self.time_mlp(t)    # [B, d_model]
        
        # Add Time Embedding to every token in the sequence 
        # (เพื่อให้ทุกจุดในกราฟรู้ว่าตอนนี้ Noise step ไหน)
        x_emb = x_emb + t_emb[:, None, :] 
        
        # --- 3. Transformer Block ---
        # Self-Attention จะทำงานตรงนี้เพื่อดูความสัมพันธ์ทั้งกราฟ
        latent = self.transformer_encoder(x_emb) # [B, L, d_model]
        
        # --- 4. Output ---
        output = self.output_proj(latent) # [B, L, C] -> Predicted Noise
        
        return output