import torch
import torch.nn as nn
from torch import Tensor, Size
from typing import Optional
import numpy as np


class DDIM(nn.Module):
    """
    DDIM (Denoising Diffusion Implicit Models) — Song et al. 2020
    https://arxiv.org/abs/2010.02502

    Drop-in replacement สำหรับ Diffusion (DDPM) ใน v4 pipeline:
      - ใช้ backbone DiffusionTransformer เหมือนเดิม (B, W, A, C)
      - Training: forward() / q_sample() เหมือน DDPM เลย (noise schedule เดิม)
      - Sampling: ใช้ DDIM non-Markovian step แทน DDPM stochastic step
        → ลด inference steps ได้มาก (1000 → 50 หรือ 100)
        → eta=0 = fully deterministic, eta=1 ≈ DDPM stochastic

    API สำคัญที่ inferencing notebook ใช้:
        model.sample(x_cond, output_shape)           → เหมือน DDPM เลย
        model.sample(x_cond, output_shape,
                     ddim_steps=50, eta=0.0)         → ปรับ steps/eta ได้

    Note: load_state_dict() ใช้ checkpoint เดิม (DDPM-trained) ได้เลย
    ไม่ต้องเทรนใหม่ — DDIM เป็น different sampler ใช้ weights เดิม
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps  # T (training timesteps)

        # ── Noise Schedule (เหมือน DDPM ทุกอย่าง) ──────────────────────
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_hat", torch.cumprod(self.alphas, dim=0))

        # Forward process variables (ใช้ทั้ง training + q_sample)
        self.register_buffer("sqrt_alphas_hat", torch.sqrt(self.alphas_hat))
        self.register_buffer(
            "sqrt_one_minus_alphas_hat", torch.sqrt(1.0 - self.alphas_hat)
        )

        # ── Reverse Variables (DDPM-style, ใช้ตอน p_sample fallback) ──
        self.register_buffer("sqrt_recip_alpha", 1.0 / torch.sqrt(self.alphas))
        self.register_buffer(
            "coef_noise", (1.0 - self.alphas) / torch.sqrt(1.0 - self.alphas_hat)
        )

        alphas_hat_prev = torch.nn.functional.pad(
            self.alphas_hat[:-1], (1, 0), value=1.0
        )
        posterior_variance = (
            self.betas * (1.0 - alphas_hat_prev) / (1.0 - self.alphas_hat)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("alphas_hat_prev", alphas_hat_prev)

    # ── Utility ──────────────────────────────────────────────────────────────

    def extract(self, a: Tensor, t: Tensor, x_shape: Size) -> Tensor:
        """Gather coefficients at timesteps t → reshape to (B, 1, 1, ...)"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        n_shapes = len(x_shape)
        return out.reshape(batch_size, *((1,) * (n_shapes - 1))).to(t.device)

    # ── Forward Process (Training, same as DDPM) ─────────────────────────────

    def q_sample(
        self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """x_0 → x_t (เหมือน DDPM เลย)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_hat_t = self.extract(self.sqrt_alphas_hat, t, x_0.shape)
        sqrt_one_minus_t = self.extract(self.sqrt_one_minus_alphas_hat, t, x_0.shape)
        return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_t * noise

    def forward(self, x_0: Tensor, x_cond: Tensor) -> Tensor:
        """Training step — เหมือน DDPM ทุกอย่าง"""
        batch_size = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        predicted_noise = self.model(x_t, t, x_cond)

        return predicted_noise, noise

    # ── DDPM Reverse Step (สำรองไว้, ไม่ได้ใช้ตอน DDIM sample) ───────────────

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, t_index: int, x_cond: Tensor) -> Tensor:
        """DDPM stochastic reverse step (ใช้ใน sample_trajectory เท่านั้น)"""
        predicted_noise = self.model(x_t, t, x_cond)

        coef_1 = self.extract(self.sqrt_recip_alpha, t, x_t.shape)
        coef_2 = self.extract(self.coef_noise, t, x_t.shape)
        model_mean = coef_1 * (x_t - coef_2 * predicted_noise)

        if t_index > 0:
            noise = torch.randn_like(x_t)
            posterior_var_t = self.extract(self.posterior_variance, t, x_t.shape)
            return model_mean + torch.sqrt(posterior_var_t) * noise
        else:
            return model_mean

    # ── DDIM Core Step ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: Tensor,  # (B, W, A, C)
        t: Tensor,  # (B,) — current timestep
        t_prev: Tensor,  # (B,) — previous (smaller) timestep
        x_cond: Tensor,  # (B, W, A, C_c)
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM reverse step: x_t → x_{t_prev}

        Algorithm (Eq. 12, Song et al. 2020):
          1. Predict noise ε_θ(x_t, t)
          2. Predict x_0 from x_t and ε  ("predicted x_0")
          3. Compute direction pointing to x_t  ("predicted direction")
          4. Combine with optional stochastic noise (controlled by eta)

        eta=0  → fully deterministic (pure DDIM)
        eta=1  → recovers DDPM-like stochasticity
        0<eta<1 → trade-off

        t_prev = -1 หมายถึง step สุดท้าย (→ x_0), ใช้ alpha_hat = 1
        """
        # ── 1. Predict noise ──────────────────────────────────────────
        eps = self.model(x_t, t, x_cond)  # (B, W, A, C)

        # ── 2. Get alpha_hat values ───────────────────────────────────
        # alpha_hat_t
        a_t = self.extract(self.alphas_hat, t, x_t.shape)  # (B,1,1,1)

        # alpha_hat_{t_prev}: ถ้า t_prev < 0 แสดงว่าถึง x_0 แล้ว → alpha_hat = 1
        # ใช้ clamp(0) เพื่อ handle กรณี t_prev = -1
        t_prev_clamped = t_prev.clamp(min=0)
        a_prev = self.extract(self.alphas_hat, t_prev_clamped, x_t.shape)  # (B,1,1,1)

        # mask สำหรับ step สุดท้าย (t_prev < 0) → override a_prev = 1
        is_last = (t_prev < 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        a_prev = a_prev * (1 - is_last) + 1.0 * is_last

        # ── 3. Predict x_0 (คาดเดา clean sample จาก x_t) ────────────
        # x_0_pred = (x_t - sqrt(1 - α̂_t) * ε) / sqrt(α̂_t)
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_1ma_t = torch.sqrt(1.0 - a_t)
        x_0_pred = (x_t - sqrt_1ma_t * eps) / sqrt_a_t.clamp(min=1e-8)

        # Optional: clip x_0_pred ให้อยู่ใน range ที่สมเหตุสมผล
        # (ช่วยกัน error explosion เหมือน clamp ใน DDPM inpainting)
        x_0_pred = x_0_pred.clamp(-10.0, 10.0)

        # ── 4. Compute sigma (stochastic noise scale) ─────────────────
        # σ_t = eta * sqrt( (1 - α̂_{t-1}) / (1 - α̂_t) * (1 - α̂_t / α̂_{t-1}) )
        sigma = eta * torch.sqrt(
            (1.0 - a_prev)
            / (1.0 - a_t).clamp(min=1e-8)
            * (1.0 - a_t / a_prev.clamp(min=1e-8))
        ).clamp(min=0.0)

        # ── 5. Direction pointing to x_t ─────────────────────────────
        # = sqrt(1 - α̂_{t-1} - σ²) * ε_θ
        coef_dir = torch.sqrt((1.0 - a_prev - sigma**2).clamp(min=0.0))
        direction = coef_dir * eps

        # ── 6. Combine: x_{t_prev} = sqrt(α̂_{t_prev}) * x_0_pred + direction + σ * z
        noise = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)
        x_prev = torch.sqrt(a_prev) * x_0_pred + direction + sigma * noise

        return x_prev

    # ── Sampling (Main API — drop-in เหมือน DDPM.sample) ─────────────────────

    @torch.no_grad()
    def sample(
        self,
        x_cond: Tensor,
        output_shape: Size,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM generation — drop-in replacement ของ DDPM.sample()

        Args:
            x_cond       : (B, W, A, C_c)  — condition (scaled, ไม่มี noise)
            output_shape : (B, W, A, C_x)  — shape ของ output ที่ต้องการ
            ddim_steps   : จำนวน denoising steps (default 50)
                           น้อยกว่า self.timesteps ได้มาก (e.g. 50 แทน 1000)
            eta          : noise scale (0 = deterministic, 1 ≈ DDPM)

        Returns:
            x_0          : (B, W, A, C_x)  — generated sample
        """
        device = x_cond.device
        B = x_cond.shape[0]

        # ── สร้าง DDIM timestep schedule ──────────────────────────────
        # เลือก ddim_steps timesteps แบบ uniformly spaced จาก [0, T-1]
        # แล้ว reverse เพื่อ sample จาก T → 0
        step_indices = torch.linspace(
            0, self.timesteps - 1, ddim_steps, dtype=torch.long
        )
        # step_indices: [t_0, t_1, ..., t_{S-1}] เรียงน้อย→มาก
        # เราจะ reverse: [t_{S-1}, ..., t_1, t_0]
        # t_prev ของแต่ละ step = step ก่อนหน้า (หรือ -1 ถ้าเป็น step แรก)

        # ── Start from pure noise x_T ─────────────────────────────────
        x = torch.randn(output_shape, device=device)

        # ── DDIM reverse loop ─────────────────────────────────────────
        # loop จาก index S-1 → 0  (คือ timestep ใหญ่ → เล็ก)
        for i in reversed(range(ddim_steps)):
            t_val = step_indices[i].item()
            t_prev_val = step_indices[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev_batch = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            x = self.ddim_step(x, t_batch, t_prev_batch, x_cond, eta=eta)

        return x

    # ── Inpainting ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inpainting_sampler(
        self,
        x: Tensor,
        cond: Tensor,
        mask: Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM Inpainting (Replacement Method) — DDIM version

        Args:
            x           : (B, W, A, C_x)  — ground truth (Known + Unknown)
            cond        : (B, W, A, C_c)  — condition features
            mask        : (B, W, A, C_x)  — 1=Keep Known, 0=Inpaint
            ddim_steps  : number of denoising steps
            eta         : noise scale

        Note: inpainting ด้วย DDIM ยังเป็น replacement method เหมือน DDPM
        แต่ใช้ DDIM step แทน p_sample → เร็วขึ้นตาม ddim_steps ratio
        """
        B = x.shape[0]
        device = x.device

        step_indices = torch.linspace(
            0, self.timesteps - 1, ddim_steps, dtype=torch.long
        )

        x_t = torch.randn_like(x)

        for i in reversed(range(ddim_steps)):
            t_val = step_indices[i].item()
            t_prev_val = step_indices[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev_batch = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            # A. DDIM step สำหรับทั้ง sequence
            x_t_pred = self.ddim_step(x_t, t_batch, t_prev_batch, cond, eta=eta)
            x_t_pred = x_t_pred.clamp(-10.0, 10.0)

            # B. Known part at t_prev noise level
            if i > 0:
                noise_known = torch.randn_like(x)
                x_t_known = self.q_sample(x, t_prev_batch, noise=noise_known)
            else:
                x_t_known = x

            # C. Replacement
            x_t = mask * x_t_known + (1 - mask) * x_t_pred

        return x_t

    # ── Trajectory (debug / visualization) ────────────────────────────────────

    @torch.no_grad()
    def sample_trajectory(
        self,
        x: Tensor,
        cond: Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> list:
        """
        Run DDIM loop แล้ว return ทุก intermediate state
        Return: [x_{t_{S-1}}, ..., x_0]  (length = ddim_steps)
        """
        B = x.shape[0]
        device = x.device

        step_indices = torch.linspace(
            0, self.timesteps - 1, ddim_steps, dtype=torch.long
        )

        x_t = torch.randn_like(x)
        trajectory = []

        for i in reversed(range(ddim_steps)):
            t_val = step_indices[i].item()
            t_prev_val = step_indices[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev_batch = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            x_t = self.ddim_step(x_t, t_batch, t_prev_batch, cond, eta=eta)
            trajectory.append(x_t.clone())

        return trajectory

    # ── Triangle Mask (inpainting helper, เหมือน DDPM) ───────────────────────

    def create_lower_right_triangle_mask(self, x: torch.Tensor):
        B, A, T, C = x.shape
        device = x.device
        base_mask = self.process_lower_right_triangle(T, C).to(device)
        return base_mask.reshape(1, 1, T, C).expand(B, A, -1, -1)

    def process_lower_right_triangle(self, length: int, feature: int):
        ones = torch.ones(length, feature)
        tril_mask = torch.tril(ones)
        return torch.flip(tril_mask, dims=[0])
