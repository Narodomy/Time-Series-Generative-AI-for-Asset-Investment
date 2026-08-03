import torch
import torch.nn as nn
from torch import Tensor, Size
from typing import Optional
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(
        self, model: nn.Module, timesteps: int, beta_start: float, beta_end: float
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # 1. Beta Schedule
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_hat", torch.cumprod(self.alphas, dim=0))

        # 2. Forward Variables
        # Use register_buffer to save these tensors with the model state_dict
        self.register_buffer("sqrt_alphas_hat", torch.sqrt(self.alphas_hat))
        self.register_buffer(
            "sqrt_one_minus_alphas_hat", torch.sqrt(1.0 - self.alphas_hat)
        )  # Renamed from sigmas for clarity

        # 3. Reverse Variables (Equation 11 Coefs)
        # Coef 1: 1/sqrt(alpha)
        self.register_buffer("sqrt_recip_alpha", 1.0 / torch.sqrt(self.alphas))

        # Coef 2: beta / sqrt(1 - alpha_hat)
        self.register_buffer(
            "coef_noise", (1.0 - self.alphas) / torch.sqrt(1.0 - self.alphas_hat)
        )

        # 4. Posterior Variance (sigma^2)
        alphas_hat_prev = torch.nn.functional.pad(
            self.alphas_hat[:-1], (1, 0), value=1.0
        )
        posterior_variance = (
            self.betas * (1.0 - alphas_hat_prev) / (1.0 - self.alphas_hat)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def extract(self, a: Tensor, t: Tensor, x_shape: Size) -> Tensor:
        """Extracts coefficients at specified timesteps t and reshapes to [B, 1, 1, ...]"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        n_shapes = len(x_shape)
        return out.reshape(batch_size, *((1,) * (n_shapes - 1))).to(t.device)

    # --- Forward Process (q_sample) ---
    def q_sample(
        self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward diffusion process: x_0 -> x_t
        Adds noise to x_0 based on timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_hat_t = self.extract(self.sqrt_alphas_hat, t, x_0.shape)
        sqrt_one_minus_alphas_hat_t = self.extract(
            self.sqrt_one_minus_alphas_hat, t, x_0.shape
        )

        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        x_t = (sqrt_alpha_hat_t * x_0) + (sqrt_one_minus_alphas_hat_t * noise)

        return x_t

    def forward(self, x_0: Tensor, x_cond: Tensor) -> Tensor:
        """
        Training Step: Calculates MSE Loss between actual noise and predicted noise.
        """

        batch_size = x_0.shape[0]
        device = x_0.device

        # 1. Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # 2. Create noise
        noise = torch.randn_like(x_0)

        # 3. Add noise to x_0 to get x_t
        x_t = self.q_sample(x_0, t, noise=noise)

        # 4. Model predicts the noise (Conditioned on x_cond)
        # Note: x_cond is clean (no noise added)
        predicted_noise = self.model(x_t, t, x_cond)

        # 5. Calculate Loss
        # loss = nn.functional.mse_loss(predicted_noise, noise)
        # return loss

        # Return these for manual calc
        return predicted_noise, noise

    # --- Reverse Process (p_sample) ---
    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, t_index: int, x_cond: Tensor) -> Tensor:
        """
        Reverse diffusion step: x_t -> x_{t-1}
        Predicts mean and adds variance noise.
        """
        # 1. Predict noise using the model
        predicted_noise = self.model(x_t, t, x_cond)

        # 2. Calculate Mean (mu)
        coef_1 = self.extract(self.sqrt_recip_alpha, t, x_t.shape)
        coef_2 = self.extract(self.coef_noise, t, x_t.shape)
        model_mean = coef_1 * (x_t - (coef_2 * predicted_noise))

        # 3. Add Variance (if not the last step)
        if t_index > 0:
            noise = torch.randn_like(x_t)
            posterior_var_t = self.extract(self.posterior_variance, t, x_t.shape)
            # x_{t-1} = mean + std * z
            return model_mean + (torch.sqrt(posterior_var_t) * noise)
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, x_cond: Tensor, output_shape: Size) -> Tensor:
        """
        Standard generation without inpainting.
        Args:
            x_cond: Conditional input [Batch, ...]
            output_shape: Shape of the target x [Batch, Channels, Length, Assets]
        """
        device = x_cond.device
        B = x_cond.shape[0]

        # Start from Pure Noise (x_T)
        x = torch.randn(output_shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, x_cond)

        return x

    # --- Inpainting Process ---
    # @torch.no_grad()
    # def inpainting_sampler(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
    #     """
    #     Sampling with Inpainting (Replacement Method)

    #     Args:
    #         x: The full target data (Known + Unknown parts)
    #                         Shape: [B, W, A, F_target]
    #         cond: Condition features (Clean, e.g., Volume, Open)
    #                 Shape: [B, W, A, F_cond]
    #         mask: Binary mask (1=Keep Known/Ground Truth, 0=Inpaint/Generate)
    #               Shape: [B, W, A, F_target]
    #     """
    #     B = x.shape[0]
    #     device = x.device

    #     # 1. Start from Pure Noise (x_T)
    #     x_t = torch.randn_like(x)

    #     # Loop backwards from T-1 to 0
    #     for i in tqdm(reversed(range(0, self.timesteps)), desc="Inpainting", total=self.timesteps, leave=False):
    #         # Current t tensor
    #         t = torch.full((B,), i, device=device, dtype=torch.long)

    #         # A. Predict x_{t-1} for the WHOLE sequence (Model's guess)
    #         x_t_minus_1_pred = self.p_sample(x_t, t, i, cond)

    #         # B. Prepare x_{t-1} for the KNOWN part
    #         if i > 0:
    #             # If not the last step, we need to add noise to the ground truth
    #             # so it matches the noise level of step t-1
    #             t_next = torch.full((B,), i - 1, device=device, dtype=torch.long)
    #             x_t_minus_1_known = self.q_sample(x, t_next)
    #         else:
    #             # At the last step (t=0), the known part is just the clean ground truth
    #             x_t_minus_1_known = x

    #         # C. Replacement Step (Combine Known & Predicted)
    #         # mask: 1 for Known, 0 for Unknown
    #         x_t = (mask * x_t_minus_1_known) + ((1 - mask) * x_t_minus_1_pred)

    #     # [B, C, L, A] shape
    #     return x_t # Final Inpainted Result (x_0)

    @torch.no_grad()
    def inpainting_sampler(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        """
        Sampling with Inpainting (Replacement Method)

        Args:
            x: The full target data (Known + Unknown parts)
                            Shape: [B, W, A, F_target]
            cond: Condition features (Clean, e.g., Volume, Open)
                    Shape: [B, W, A, F_cond]
            mask: Binary mask (1=Keep Known/Ground Truth, 0=Inpaint/Generate)
                  Shape: [B, W, A, F_target]
        """
        B = x.shape[0]
        device = x.device

        # 1. Start from Pure Noise (x_T)
        x_t = torch.randn_like(x)

        # Loop backwards from T-1 to 0
        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="Inpainting",
            total=self.timesteps,
            leave=False,
        ):
            # Current t tensor
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # A. Predict x_{t-1} for the WHOLE sequence (Model's guess)
            x_t_minus_1_pred = self.p_sample(x_t, t, i, cond)

            # 🌟 FIX: ใส่สมอถ่วงไว้ ไม่ให้ Transformer พ่นค่าระเบิดเกิน +- 1.0 (Standard Normal)
            x_t_minus_1_pred = torch.clamp(x_t_minus_1_pred, min=-10.0, max=10.0)

            # B. Prepare x_{t-1} for the KNOWN part
            if i > 0:
                t_next = torch.full((B,), i - 1, device=device, dtype=torch.long)
                x_t_minus_1_known = self.q_sample(x, t_next)
            else:
                x_t_minus_1_known = x

            # C. Replacement Step
            x_t = (mask * x_t_minus_1_known) + ((1 - mask) * x_t_minus_1_pred)

        # [B, C, L, A] shape
        return x_t  # Final Inpainted Result (x_0)

    # @torch.no_grad()
    # def inpainting_sampler(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
    #     """
    #     Sampling with Inpainting (Replacement Method)

    #     Args:
    #         x: The full target data (Known + Unknown parts)
    #                         Shape: [B, W, A, F_target]
    #         cond: Condition features (Clean, e.g., Volume, Open)
    #                 Shape: [B, W, A, F_cond]
    #         mask: Binary mask (1=Keep Known/Ground Truth, 0=Inpaint/Generate)
    #               Shape: [B, W, A, F_target]
    #     """
    #     B = x.shape[0]
    #     device = x.device

    #     # 1. Start from Pure Noise (x_T)
    #     x_t = torch.randn_like(x)

    #     # Loop backwards from T-1 to 0
    #     for i in tqdm(reversed(range(0, self.timesteps)), desc="Inpainting", total=self.timesteps, leave=False):
    #         t = torch.full((B,), i, device=device, dtype=torch.long)

    #         # A. Predict x_{t-1}
    #         x_t_minus_1_pred = self.p_sample(x_t, t, i, cond)

    #         # 🌟 จุดที่ 1: Dynamic Clipping หรือ Safety Clamp
    #         # ถ้า x_t_minus_1_pred คือตัวที่มี Noise การ Clip แน่นเกินไป ([-1,1]) จะทำให้ภาพดู "Burn" หรือ Contrast จัดเกินไป
    #         # แนะนำให้ใช้ค่าที่เผื่อ Noise ไว้หน่อย เช่น [-2, 2] หรือใช้ [-10, 10] แบบเดิมก็ได้เพื่อความปลอดภัย
    #         x_t_minus_1_pred = torch.clamp(x_t_minus_1_pred, min=-10.0, max=10.0)

    #         # B. Prepare Known Part
    #         if i > 0:
    #             t_next = torch.full((B,), i - 1, device=device, dtype=torch.long)
    #             x_t_minus_1_known = self.q_sample(x, t_next)
    #         else:
    #             x_t_minus_1_known = x # ขั้นตอนสุดท้ายใช้ของจริงเลย

    #         # C. Replacement Step
    #         x_t = (mask * x_t_minus_1_known) + ((1 - mask) * x_t_minus_1_pred)

    #     # 🌟 จุดที่ 2: Final Output Clipping (สำคัญมาก)
    #     # ก่อนส่งออกไปใช้งาน ต้องมั่นใจว่าค่าอยู่ใน Range [-1, 1] จริงๆ
    #     x_t = torch.clamp(x_t, min=-1.0, max=1.0)

    #     return x_t

    # # --- Inpainting Process (No Stitching Version) ---
    # @torch.no_grad()
    # def inpainting_sampler(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
    #     """
    #     Sampling with 'Guide & Replace at the end' Method
    #     (เหมาะสมที่สุดสำหรับ Transformer เพื่อป้องกันการเกิด Frankenstein Tensor)
    #     """
    #     B = x.shape[0]
    #     device = x.device

    #     # 1. เริ่มต้นจาก Pure Noise (x_T)
    #     x_t = torch.randn_like(x)

    #     # 2. วนลูป Denoise จาก T-1 ถึง 0 (ปล่อยให้โมเดล Gen แบบธรรมชาติ)
    #     for i in tqdm(reversed(range(0, self.timesteps)), desc="Generating", total=self.timesteps, leave=False):
    #         t = torch.full((B,), i, device=device, dtype=torch.long)

    #         # ให้โมเดลทำนาย step ก่อนหน้า (x_{t-1}) โดยมี cond คอยไกด์
    #         x_t = self.p_sample(x_t, t, i, cond)

    #         # 🛡️ ใส่เบรก (Clamp) ป้องกันการสโนว์บอลของ Error เผื่อโมเดลสะบัด
    #         # ปกติ Standard Normal จะแกว่งแค่ช่วง -3 ถึง 3 การคุมไว้ที่ +- 5 จะช่วยล็อกสเกลให้ปลอดภัย
    #         x_t = torch.clamp(x_t, min=-10.0, max=10.0)

    #     # 3. เย็บตะเข็บครั้งเดียวตอนจบ! (Final Replacement)
    #     # ถึงตรงนี้ x_t คือ x_0 ที่ถูก Denoise จนใสปิ๊งแล้ว
    #     # เราเอาข้อมูลอดีตของจริง (x) มาทับเฉพาะส่วนที่ Mask เป็น 1
    #     x_final = (mask * x) + ((1 - mask) * x_t)

    #     # [B, C, L, A] shape
    #     return x_final

    def create_lower_right_triangle_mask(self, x: torch.Tensor):
        # param x shape: [Batch, Channel, Length, Assets]
        B, C, L, A = x.shape
        device = x.device

        # example: L=3, C=3
        # 1 1 1
        # 1 1 0
        # 1 0 0
        base_mask = self.process_lower_right_triangle(L, C).to(device)

        # Adjust Shape match x
        # base_mask = [L, C] but x sort to [..., C, L, ...]
        # Transpose [C, L] first
        base_mask = base_mask.transpose(0, 1)  # ได้ shape [C, L]

        # Broadcast
        # เ[C, L] -> [1, C, L, 1]
        final_mask = base_mask.reshape(1, C, L, 1)

        full_mask = final_mask.expand(B, -1, -1, A)

        return full_mask

    def process_lower_right_triangle(self, length: int, feature: int):
        ones = torch.ones(length, feature)

        # Lower Triangle
        # 1 0 0
        # 1 1 0
        # 1 1 1
        tril_mask = torch.tril(ones)

        # Flip Up-Down
        # 1 1 1
        # 1 1 0
        # 1 0 0

        final_mask = torch.flip(tril_mask, dims=[0])

        return final_mask

    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor, cond: torch.Tensor) -> list:
        """
        Run the full reverse diffusion loop and return every intermediate state.
        Return order: [x_T, x_{T-1}, ..., x_0]
        Compatible with DDPM now; swap the denoising step later for DDIM.
        """
        trajectory = []
        x_t = torch.randn_like(x)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, t, cond)  # ← swap to DDIM step later
            trajectory.append(x_t.clone())

        return trajectory  # length = T, ordered x_{T-1}...x_0 (already T→0)
