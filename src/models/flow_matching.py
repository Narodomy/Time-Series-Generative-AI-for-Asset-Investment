import torch
import torch.nn as nn
from torch import Tensor, Size
from typing import Optional
from tqdm import tqdm


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching (CFM) — Lipman et al. 2022 / Liu et al. 2022
    https://arxiv.org/abs/2210.02747  (Flow Matching for Generative Modeling)
    https://arxiv.org/abs/2209.03003  (Flow Straight and Fast: Rectified Flow)

    Drop-in replacement สำหรับ Diffusion / DDIM ใน v4 pipeline:
      - ใช้ backbone DiffusionTransformer เหมือนเดิม (B, W, A, C)
      - Training: เรียนรู้ vector field  u_θ(x_t, t, cond)
                  แทน noise ε_θ ใน DDPM
      - Sampling: integrate ODE  dx = u_θ(x_t, t, cond) dt
                  ด้วย simple Euler หรือ Midpoint (ต้องการ steps น้อยกว่า DDPM มาก)

    ─── Probability Path ──────────────────────────────────────────────────────
    ใช้ Optimal Transport (OT) conditional path:
        x_t = (1 - t) * x_0 + t * x_1      ,  t ∈ [0, 1]

    โดยที่:
        x_0 ~ N(0, I)   (noise / source)
        x_1 ~ p_data    (clean data / target)

    Vector field ที่ต้องเรียนรู้:
        u*(x_t | x_0, x_1) = x_1 - x_0     (constant along each path)

    Loss:
        L = E_{t, x_0, x_1} [ || u_θ(x_t, t, cond) - (x_1 - x_0) ||² ]

    ─── Sampling ──────────────────────────────────────────────────────────────
    Euler:    x_{t + dt} = x_t + u_θ(x_t, t, cond) * dt
    Midpoint: x_mid = x_t + u_θ(x_t, t) * (dt/2)
              x_{t+dt} = x_t + u_θ(x_mid, t + dt/2) * dt

    ─── API ───────────────────────────────────────────────────────────────────
    model.forward(x_1, x_cond)                          → (v_pred, v_target) สำหรับ training
    model.sample(x_cond, output_shape)                  → x_1  (Euler, 100 steps)
    model.sample(x_cond, output_shape,
                 num_steps=50, method="midpoint")       → x_1
    model.inpainting_sampler(x, cond, mask)             → x_1
    model.sample_trajectory(x, cond, num_steps=20)     → list[Tensor]
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4,
    ):
        """
        Args:
            model     : backbone (DiffusionTransformer หรือ FlowMatchingTransformer)
                        signature: model(x_t, t, cond) → vector field prediction
                        *หมายเหตุ* t ที่ส่งเข้าโมเดลเป็น float tensor ใน [0,1]
                        ถ้าใช้ SinusoidalPositionEmbeddings แบบ DDPM (รับ int timestep)
                        ให้ scale ก่อน: t_int = (t * 999).long()
                        ดู _t_to_model() ด้านล่าง
            sigma_min : minimum noise std ที่ target path (ช่วย numerical stability)
                        ใช้ค่าเล็กมากๆ เช่น 1e-4 ก็พอ (ตาม paper)
        """
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    # ── Utility ──────────────────────────────────────────────────────────────

    def _t_to_model(self, t: Tensor) -> Tensor:
        """
        แปลง continuous t ∈ [0, 1] → int timestep ที่โมเดลรับได้

        DiffusionTransformer ใช้ SinusoidalPositionEmbeddings ที่ออกแบบมาสำหรับ
        discrete DDPM timestep (0 … T-1)  ดังนั้นเราแปลง t → int แทนที่จะแก้ backbone

        ถ้าเปลี่ยนไปใช้ backbone ใหม่ที่รับ float ได้โดยตรง ให้ return t เลย
        """
        t_int = (t * 999).long().clamp(0, 999)  # map [0,1] → [0, 999]
        return t_int

    def _get_xt(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Interpolate ตาม OT path:  x_t = (1 - t) * x_0 + t * x_1

        พร้อม sigma_min perturbation ที่ target เพื่อ numerical stability:
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        """
        # Reshape t → (B, 1, 1, 1) เพื่อ broadcast กับ (B, W, A, C)
        n_dims = x_1.dim()
        t_b = t.view(-1, *([1] * (n_dims - 1)))

        x_t = (1 - (1 - self.sigma_min) * t_b) * x_0 + t_b * x_1
        return x_t

    def _get_target_v(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """
        Target vector field:  u* = x_1 - (1 - sigma_min) * x_0
        (derivative ของ path ตาม t, constant ตลอด path)
        """
        return x_1 - (1 - self.sigma_min) * x_0

    # ── Training ─────────────────────────────────────────────────────────────

    def forward(self, x_1: Tensor, x_cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Training step — คืน (v_pred, v_target) สำหรับคำนวณ loss ข้างนอก
        (pattern เดียวกับ DDPM.forward ที่คืน predicted_noise, noise)

        Args:
            x_1    : (B, W, A, C)     — clean data (target)
            x_cond : (B, W, A, C_c)   — conditioning features

        Returns:
            v_pred   : (B, W, A, C)   — predicted vector field
            v_target : (B, W, A, C)   — target vector field (ground truth)
        """
        B = x_1.shape[0]
        device = x_1.device

        # 1. Sample t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)

        # 2. Sample source noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # 3. Interpolate → x_t
        x_t = self._get_xt(x_0, x_1, t)

        # 4. Compute target vector field
        v_target = self._get_target_v(x_0, x_1)

        # 5. Model predicts vector field
        t_model = self._t_to_model(t)
        v_pred = self.model(x_t, t_model, x_cond)

        return v_pred, v_target

    # ── ODE Solvers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _euler_step(
        self,
        x_t: Tensor,
        t: float,
        dt: float,
        x_cond: Tensor,
    ) -> Tensor:
        """Euler integration step: x_{t+dt} = x_t + v_θ(x_t, t) * dt"""
        B = x_t.shape[0]
        device = x_t.device

        t_batch = torch.full((B,), t, device=device)
        t_model = self._t_to_model(t_batch)

        v = self.model(x_t, t_model, x_cond)
        v = v.clamp(-10.0, 10.0)  # safety clamp เหมือน DDPM

        return x_t + v * dt

    @torch.no_grad()
    def _midpoint_step(
        self,
        x_t: Tensor,
        t: float,
        dt: float,
        x_cond: Tensor,
    ) -> Tensor:
        """
        Midpoint (RK2) step — ต้องการ 2x model calls ต่อ step แต่ accurate กว่า Euler
        เหมาะสำหรับ num_steps น้อย (เช่น 20-50)
        """
        B = x_t.shape[0]
        device = x_t.device

        # ── Half step (evaluate at midpoint) ────────────────────────────
        t_batch = torch.full((B,), t, device=device)
        t_model = self._t_to_model(t_batch)
        v1 = self.model(x_t, t_model, x_cond).clamp(-10.0, 10.0)

        x_mid = x_t + v1 * (dt / 2)
        t_mid = t + dt / 2

        # ── Full step (evaluate at midpoint, apply full dt) ──────────────
        t_mid_batch = torch.full((B,), t_mid, device=device).clamp(0.0, 1.0)
        t_mid_model = self._t_to_model(t_mid_batch)
        v2 = self.model(x_mid, t_mid_model, x_cond).clamp(-10.0, 10.0)

        return x_t + v2 * dt

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        x_cond: Tensor,
        output_shape: Size,
        num_steps: int = 100,
        method: str = "euler",
    ) -> Tensor:
        """
        ODE integration จาก t=0 (noise) → t=1 (data)

        Args:
            x_cond       : (B, W, A, C_c) — conditioning features
            output_shape : (B, W, A, C_x) — desired output shape
            num_steps    : จำนวน integration steps (น้อยกว่า DDPM มาก: 20-100 พอ)
            method       : "euler" | "midpoint"

        Returns:
            x_1 : (B, W, A, C_x) — generated sample
        """
        device = x_cond.device

        # Start from pure noise at t=0
        x = torch.randn(output_shape, device=device)

        dt = 1.0 / num_steps
        t_schedule = torch.linspace(0.0, 1.0 - dt, num_steps)  # [t_0, ..., t_{N-1}]

        step_fn = self._midpoint_step if method == "midpoint" else self._euler_step

        for t in t_schedule:
            x = step_fn(x, t.item(), dt, x_cond)

        return x

    # ── Inpainting ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inpainting_sampler(
        self,
        x: Tensor,
        cond: Tensor,
        mask: Tensor,
        num_steps: int = 100,
        method: str = "euler",
    ) -> Tensor:
        """
        Flow Matching Inpainting (Replacement Method)

        ใช้ pattern เดียวกับ DDPM inpainting:
            - Known region : ใช้ noisy version ของ ground truth ที่ t ปัจจุบัน
            - Unknown region: ใช้ model prediction

        Args:
            x        : (B, W, A, C_x) — ground truth (Known + Unknown)
            cond     : (B, W, A, C_c) — conditioning features
            mask     : (B, W, A, C_x) — 1=Keep Known, 0=Inpaint
            num_steps: integration steps
            method   : "euler" | "midpoint"

        Returns:
            x_1 : (B, W, A, C_x) — inpainted result
        """
        B = x.shape[0]
        device = x.device

        # Source noise (ใช้ fixed noise เพื่อ consistent inpainting)
        x_0_known = torch.randn_like(x)

        # Start from pure noise at t=0
        x_t = torch.randn_like(x)

        dt = 1.0 / num_steps
        t_schedule = torch.linspace(0.0, 1.0 - dt, num_steps)

        step_fn = self._midpoint_step if method == "midpoint" else self._euler_step

        for t in t_schedule:
            t_val = t.item()

            # A. Model step สำหรับทั้ง sequence
            x_t_pred = step_fn(x_t, t_val, dt, cond)

            # B. Known part: interpolate ground truth ที่ t+dt
            #    x_t+dt_known = (1 - (1 - σ_min)(t+dt)) * x_0 + (t+dt) * x_1
            t_next = min(t_val + dt, 1.0)
            n_dims = x.dim()
            t_next_b = torch.full((B, *([1] * (n_dims - 1))), t_next, device=device)
            x_t_known = (1 - (1 - self.sigma_min) * t_next_b) * x_0_known + t_next_b * x

            # C. Replacement
            x_t = mask * x_t_known + (1 - mask) * x_t_pred

        return x_t

    # ── Trajectory (debug / visualization) ────────────────────────────────────

    @torch.no_grad()
    def sample_trajectory(
        self,
        x: Tensor,
        cond: Tensor,
        num_steps: int = 20,
        method: str = "euler",
    ) -> list[Tensor]:
        """
        Run FM ODE loop แล้ว return ทุก intermediate state

        Returns:
            trajectory : list[Tensor] ความยาว num_steps
                         [x_t0, x_t1, ..., x_1]  (t เพิ่มขึ้น 0 → 1)
        """
        device = cond.device
        output_shape = x.shape

        x_t = torch.randn(output_shape, device=device)
        trajectory = []

        dt = 1.0 / num_steps
        t_schedule = torch.linspace(0.0, 1.0 - dt, num_steps)

        step_fn = self._midpoint_step if method == "midpoint" else self._euler_step

        for t in t_schedule:
            x_t = step_fn(x_t, t.item(), dt, cond)
            trajectory.append(x_t.clone())

        return trajectory  # length = num_steps, เรียง t_0 → t_1 (น้อย → มาก)

    # ── Triangle Mask (inpainting helper, เหมือน DDPM/DDIM) ─────────────────

    def create_lower_right_triangle_mask(self, x: torch.Tensor):
        B, A, T, C = x.shape
        device = x.device
        base_mask = self.process_lower_right_triangle(T, C).to(device)
        return base_mask.reshape(1, 1, T, C).expand(B, A, -1, -1)

    def process_lower_right_triangle(self, length: int, feature: int):
        ones = torch.ones(length, feature)
        tril_mask = torch.tril(ones)
        return torch.flip(tril_mask, dims=[0])
