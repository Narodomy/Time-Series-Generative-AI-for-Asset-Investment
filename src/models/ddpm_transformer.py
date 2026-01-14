import torch
import torch.nn as nn

from .modules import SinusoidalPositionEmbeddings

class DiffusionTransformer(nn.Module):
    def __init__(
        self, 
        features_in, 
        d_model=128,    # Deep layers
        nhead=4, 
        num_layers=4, 
        dropout=0.1, 
        max_len=64     # Window Size (Max L)
    ):
        super().__init__()
        self.name = "DiffusionTransformer"
        self.d_model = d_model
        
        # Input Projection: Transform original Feature (F) has size Model (d_model)
        self.input_proj = nn.Linear(features_in, d_model)
        
        # Time Embedding (for t)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Positional Encoding (for Sequence)
        # Parameter which can use both learnable, or Sinusoidal
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # 4. Transformer Encoder
        # batch_first=True  because data shape = [Batch, Length, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True,
            activation="gelu",
            norm_first=True    # Pre-Norm is stable for Deep Networks
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Block
        self.final_norm = nn.LayerNorm(d_model) # Pre Normalize before Output
        self.output_proj = nn.Linear(d_model, features_in)

    def forward(self, x, t):
        """
        x shape: [Batch, Length, Features]
        t shape: [Batch]
        """
        B, L, C = x.shape
        
        # Input
        x_emb = self.input_proj(x)  # [B, L, d_model]
        
        # + Positional Encoding
        x_emb = x_emb + self.pos_encoder[:, :L, :]
        
        # Time Process
        t_emb = self.time_mlp(t)    # [B, d_model]
        
        # Add Time Embedding
        # let t pluse every points in the Sequence for tell the context that "all time series has Noise at t"
        x_emb = x_emb + t_emb[:, None, :] 
        
        # Transformer
        latent = self.transformer_encoder(x_emb) # [B, L, d_model]
        
        latent = self.final_norm(latent)
        output = self.output_proj(latent) # [B, L, F]
        
        return output