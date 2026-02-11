import torch
import torch.nn as nn

from .modules import SinusoidalPositionEmbeddings

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        n_features: int,      
        n_cond: int,          
        window_size: int,     
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()

        # Time (t)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Input Projection (Include Noise + Condition)
        input_dim = n_features + n_cond
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional Encoding for Seq 1...W
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))

        # Transformer Backbone (Encoder Only)
        # batch_first=True for input shape like this [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Projection 
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x_t, t, x_cond):
        """
        x_t:    [Batch, Window, Features]  <- Noise
        t:      [Batch]                    <- Time Index
        x_cond: [Batch, Window, Cond_Feats] <- Condition
        """
        # Time Embedding
        # t_emb shape: [Batch, d_model]
        t_emb = self.time_mlp(t)

        # Input (Concatenation)
        # x_input shape: [Batch, Window, Features + Cond_Feats]
        x_input = torch.cat([x_t, x_cond], dim=-1)
         
        # Project to d_model
        # x shape: [Batch, Window, d_model]
        x = self.input_proj(x_input)

        B, W, F = x.shape
        # Include Time Embedding and Positional Embedding
        # t_emb up size to Window (Broadcasting)
        # [Batch, 1, d_model] + [1, Window, d_model]
        x = x + t_emb.unsqueeze(1) + self.pos_embedding[:, :W, :] # Edited here
        # x = x + t_emb.unsqueeze(1) + self.pos_embedding

        # Transformer
        # x shape: [Batch, Window, d_model]
        x = self.transformer(x)

        # Predicted Noise
        # output shape: [Batch, Window, Features]
        return self.output_proj(x)
        
        