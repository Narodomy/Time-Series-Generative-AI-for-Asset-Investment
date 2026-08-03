import torch
import torch.nn as nn
from .modules import SinusoidalPositionEmbeddings


class AxialDecoderLayer(nn.Module):
    """
    One axial block = AssetBlock (attend over A) → TimeBlock (attend over W).

    Each sub-block internally is: self-attn(x,x) → cross-attn(x,cond) → FFN,
    matching the structure of nn.TransformerDecoderLayer, but applied along
    a single axis at a time so the two axes don't get mixed into one
    undifferentiated sequence.

    AssetBlock : no causal mask — assets at the same timestep can all see
                 each other (no natural ordering across assets).
    TimeBlock  : causal mask — timestep t cannot see t+1, t+2, ... .
    """

    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.asset_block = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.time_block = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        h_x: torch.Tensor,  # (B, W, A, d_model)
        h_cond: torch.Tensor,  # (B, W, A, d_model)
        time_causal_mask: torch.Tensor,  # (W, W)
    ) -> torch.Tensor:
        B, W, A, D = h_x.shape

        # ── AssetBlock: attend across A, independently per (B, W) ──────
        # (B, W, A, D) → (B*W, A, D)
        x_a = h_x.reshape(B * W, A, D)
        cond_a = h_cond.reshape(B * W, A, D)

        x_a = self.asset_block(
            tgt=x_a,
            memory=cond_a,
            tgt_mask=None,  # no causal mask across assets
            memory_mask=None,
        )  # (B*W, A, D)

        h_x = x_a.reshape(B, W, A, D)

        # ── TimeBlock: attend across W, independently per (B, A), causal ─
        # (B, W, A, D) → (B, A, W, D) → (B*A, W, D)
        x_t = h_x.permute(0, 2, 1, 3).reshape(B * A, W, D)
        cond_t = h_cond.permute(0, 2, 1, 3).reshape(B * A, W, D)

        x_t = self.time_block(
            tgt=x_t,
            memory=cond_t,
            tgt_mask=time_causal_mask,
            memory_mask=time_causal_mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )  # (B*A, W, D)

        # (B*A, W, D) → (B, A, W, D) → (B, W, A, D)
        h_x = x_t.reshape(B, A, W, D).permute(0, 2, 1, 3)

        return h_x


class DiffusionTransformer(nn.Module):
    """
    Axial (factorized) version: instead of flattening (W, A) -> S=W*A and
    running one big attention over all W*A tokens, each layer factorizes
    attention into:
        1. AssetBlock — cross-asset attention within a fixed timestep
        2. TimeBlock  — cross-time attention within a fixed asset (causal)

    Both x and cond stay 4D (B, W, A, C) end-to-end. No (W, A) -> S flatten
    anywhere in this model.
    """

    def __init__(
        self,
        input_dim: int,  # C_x — per-token feature dim of x
        cond_dim: int,  # C_c — per-token feature dim of cond
        d_model: int,
        num_layers: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_w: int = 128,
        max_a: int = 128,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.max_w = max_w
        self.max_a = max_a

        # ── Projections ────────────────────────────────────────
        self.x_proj = nn.Linear(input_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # ── Time embedding (diffusion timestep, not sequence time) ──
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Positional embeddings — separate for W and A axes ──────
        self.pos_W = nn.Parameter(torch.zeros(1, max_w, 1, d_model))
        self.pos_A = nn.Parameter(torch.zeros(1, 1, max_a, d_model))

        # ── Axial decoder layers ────────────────────────────────
        self.layers = nn.ModuleList(
            [
                AxialDecoderLayer(
                    d_model=d_model,
                    num_attention_heads=num_attention_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ── Output projection ──────────────────────────────────
        self.output_proj = nn.Linear(d_model, input_dim)

        # Zero-init for training stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,  # (B, W, A, C_x)
        t: torch.Tensor,  # (B,) — diffusion timestep
        cond: torch.Tensor,  # (B, W, A, C_c)
    ) -> torch.Tensor:

        B, W, A, _ = x.shape
        assert W <= self.max_w, f"W={W} exceeds max_w={self.max_w}"
        assert A <= self.max_a, f"A={A} exceeds max_a={self.max_a}"

        h_x = self.x_proj(x)  # (B, W, A, d_model)
        h_cond = self.cond_proj(cond)  # (B, W, A, d_model)

        # Diffusion-timestep embedding broadcast over (W, A)
        t_emb = self.time_mlp(t)  # (B, d_model)
        h_x = h_x + t_emb[:, None, None, :]  # (B, W, A, d_model)

        # Positional embeddings — broadcast-add both axes
        pos = self.pos_W[:, :W, :, :] + self.pos_A[:, :, :A, :]  # (1, W, A, d_model)
        h_x = h_x + pos
        h_cond = h_cond + pos

        time_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            W, device=x.device
        )

        for layer in self.layers:
            h_x = layer(h_x, h_cond, time_causal_mask)

        out = self.output_proj(h_x)  # (B, W, A, C_x)

        return out
