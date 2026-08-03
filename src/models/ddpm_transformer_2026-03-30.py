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
        self.input_dim_x = (
            num_assets * num_channels
        )  # A*(Channels_target + Channels_condition)
        print(f"input dim x: {self.input_dim_x}, d_model: {d_model}")
        self.x_proj = nn.Linear(
            self.input_dim_x, d_model
        )  # Projection [Batch, Seq_length, d_model]

        # Condition
        self.input_dim_cond = num_assets * num_cond_channels
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
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

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

        self.output_dim = (
            num_assets * num_channels
        )  # we just don't need condition include from prediction!
        self.output_proj = nn.Linear(d_model, self.output_dim)

        # (Optional) Zero-init output projection for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [Batch, C_t, Time, Assets],
        x_cond: [Batch, C_cond, Time, Assets],
        t: [Batch]
        """
        # x shape: [B, C, L, A]
        B, C, L, A = x.shape

        # Flatten
        # Feature
        h_x = x.permute(0, 2, 3, 1)  # [B, L, A, C]
        h_x = h_x.reshape(B, L, -1)  # [B, L, A * C] -> [B, L, Feature_X]
        h_x = self.x_proj(h_x)  # [B, L, d_model]

        # Condition
        h_cond = cond.permute(0, 2, 3, 1)  # [B, L, A, C]
        h_cond = h_cond.reshape(B, L, -1)  # [B, L, A * C] -> [B, L, Cond]
        h_cond = self.cond_proj(h_cond)  # [B, L, d_model]

        # Embedding
        t_emb = self.time_mlp(
            t
        )  # [B, d_model] put fourier transform to say that have noise at t step time! (normaly, it's linear and adjust littke non-linear with GeLU but not much to say that have much noise!)
        h_x = h_x + t_emb.unsqueeze(1)  # from linear y = (x * W_x) + (t_emb) + b

        # Positional Embedding
        h_x = h_x + self.pos_embedding[:, :L, :]
        h_cond = h_cond + self.pos_embedding[:, :L, :]

        # Attention
        # tgt = h_x (Q,K,V in self Attn then Q in Cross-Attn)
        # memory = h_cond (K,V in Cross Attn)
        # Flow TransformerDecodeLayer
        # 1. h_x = SelfAttention(h_x, h_x, h_x)
        # 2. h_x = CrossAttention(h_x, h_cond, h_cond)
        # 3. h_x = FeedForward(h_x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)

        out_h = self.transformer(
            tgt=h_x, memory=h_cond, tgt_mask=None, memory_mask=causal_mask
        )

        # Output
        out = self.output_proj(out_h)
        out = out.reshape(B, L, A, self.num_channels).permute(
            0, 3, 1, 2
        )  # [B, C, L, A]

        return out
