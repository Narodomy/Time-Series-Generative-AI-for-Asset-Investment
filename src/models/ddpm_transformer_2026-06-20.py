import torch
import torch.nn as nn
from .modules import SinusoidalPositionEmbeddings


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,  # feat_x   = prod(feat_dims sizes)  e.g. W*D*C
        cond_dim: int,  # feat_cond = prod(cond feat sizes)  e.g. W*F
        d_model: int,
        num_layers: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # ── Projections ────────────────────────────────────────
        self.x_proj = nn.Linear(input_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # ── Time embedding ─────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Positional embedding  (learned, max 1024 seq) ──────
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))

        # ── Transformer decoder ────────────────────────────────
        # Self-Attn(x,x) → Cross-Attn(x, cond) → FFN
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ── Output projection ──────────────────────────────────
        self.output_proj = nn.Linear(d_model, input_dim)

        # Zero-init for training stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,  # (B, seq, feat_x)    — reshaped by Engine
        t: torch.Tensor,  # (B,)                — diffusion timestep
        cond: torch.Tensor,  # (B, seq, feat_cond) — reshaped by Engine
    ) -> torch.Tensor:

        B, S, _ = x.shape

        h_x = self.x_proj(x)  # (B, S, d_model)
        h_cond = self.cond_proj(cond)  # (B, S, d_model)

        # Time embedding broadcast over seq
        t_emb = self.time_mlp(t)  # (B, d_model)
        h_x = h_x + t_emb.unsqueeze(1)  # (B, S, d_model)

        # Positional embedding
        h_x = h_x + self.pos_embedding[:, :S, :]
        h_cond = h_cond + self.pos_embedding[:, :S, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        out_h = self.transformer(
            tgt=h_x,
            memory=h_cond,
            tgt_mask=causal_mask,  # self-attn mask
            memory_mask=causal_mask,  # cross-attn mask
            tgt_is_causal=True,
            memory_is_causal=True,
        )  # (B, S, d_model)
        out = self.output_proj(out_h)  # (B, S, feat_x)

        return out
