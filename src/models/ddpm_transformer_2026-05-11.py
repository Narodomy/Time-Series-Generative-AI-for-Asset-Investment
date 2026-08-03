import torch
import torch.nn as nn
import math
from typing import Optional
from .modules import SinusoidalPositionEmbeddings


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_assets: int,
        num_channels: int,
        num_cond_channels: int,
        num_layers: int,
        num_attention_heads: int,
        seq_length: int,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.num_channels = num_channels
        self.d_model = d_model

        # [B, C, T, A]
        # [B, T, A*C] -> [B, Seq_length, d_in]
        # Feature
        self.input_dim_x = seq_length * num_channels  # T*C
        print(f"input dim x: {self.input_dim_x}, d_model: {d_model}")
        self.x_proj = nn.Linear(
            self.input_dim_x, d_model
        )  # Projection [Batch, Seq_length, d_model]

        # Condition
        self.input_dim_cond = seq_length * num_cond_channels  # T*C_cond
        self.cond_proj = nn.Linear(
            self.input_dim_cond, d_model
        )  # Projection [Batch, Seq_length, d_model]

        # Embeddings
        # Time steps embedding (DDPM step: t)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Sequence PE (Day 1...N)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_assets, d_model))  # seq=A

        # Attention
        # Self-Attn (x,x) -> Cross-Attn (x, cond) -> FFN
        decoder_layer = nn.TransformerDecoderLayer(
            nhead=num_attention_heads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # PreNorm From Output = LayerNorm(x + Sublayer(x)) to this Output = x + Sublayer(LayerNorm(x))
        )

        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_dim = seq_length * num_channels  # T*C
        self.output_proj = nn.Linear(d_model, self.output_dim)

        # (Optional) Zero-init output projection for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x:    [B, A, T, C]        — noisy target (assets, window, channels)
        cond: [B, A, T, C_cond]   — clean condition (indicators)
        t:    [B]                  — diffusion timestep
        """
        B, A, T, C = x.shape

        # Flatten T*C along feature dim, keep A as sequence axis
        # [B, A, T, C] → [B, A, T*C]
        h_x = x.permute(0, 1, 2, 3).reshape(B, A, T * C)  # [B, A, T*C]
        h_x = self.x_proj(h_x)  # [B, A, d_model]

        _, _, _, C_cond = cond.shape
        h_cond = cond.reshape(B, A, T * C_cond)  # [B, A, T*C_cond]
        h_cond = self.cond_proj(h_cond)  # [B, A, d_model]

        # Time embedding
        t_emb = self.time_mlp(t)  # [B, d_model]
        h_x = h_x + t_emb.unsqueeze(1)  # broadcast over A

        # Positional embedding
        h_x = h_x + self.pos_embedding[:, :A, :]
        h_cond = h_cond + self.pos_embedding[:, :A, :]

        # Transformer Decoder
        # 1. SelfAttention(h_x, h_x, h_x)
        # 2. CrossAttention(h_x, h_cond, h_cond)
        # 3. FFN
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        out_h = self.transformer(
            tgt=h_x, memory=h_cond, tgt_mask=None, memory_mask=None
        )  # [B, A, d_model]

        # out_h = self.transformer(
        #     tgt=h_x, memory=h_cond, tgt_mask=causal_mask, memory_mask=causal_mask
        # )  # [B, T, d_model]

        # Project back and restore original layout
        out = self.output_proj(out_h)  # [B, A, T*C]
        out = out.reshape(B, A, T, C)  # [B, A, T, C]

        return out
