import torch
import torch.nn as nn
from torch import Tensor, Size
from typing import Optional
from tqdm import tqdm

class Diffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int, beta_start: float, beta_end: float):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # 1. Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        # 2. Forward Variables
        self.register_buffer('sqrt_alphas_hat', torch.sqrt(self.alphas_hat))
        self.register_buffer('sigmas', torch.sqrt(1. - self.alphas_hat))

        # 3. Reverse Variables (Equation 11 Coefs)
        # Coef 1: 1/sqrt(alpha)
        self.register_buffer('sqrt_recip_alpha', 1. / torch.sqrt(self.alphas))
        
        # Coef 2: beta / sqrt(1 - alpha_hat)
        # Noted: (1 - self.alphas) is beta
        self.register_buffer('coef_noise', (1. - self.alphas) / torch.sqrt(1. - self.alphas_hat))
        
        # 4. Posterior Variance (sigma^2)
        # Calc Variance eq
        alphas_hat_prev = torch.nn.functional.pad(self.alphas_hat[:-1], (1, 0), value=1.0)
        posterior_variance = self.betas * (1. - alphas_hat_prev) / (1. - self.alphas_hat)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def extract(self, a: Tensor, t: Tensor, x_shape: Size) -> Tensor:
        """
        Args:
            a = Alpha
            t = Time indices
            x_shape = x_0 shape
        """
        # x shape should be [B, W, A, F]
        batch_size = t.shape[0]
        out = a.gather(-1, t) # Got only shape like this  [B]
        # out = a.gather(-1, t.cpu()) # Got only shape like this  [B]

        # Reshape [B, 1 ,1 ,1]
        # To mutiply with x shape: [B, W, A, F]
        n_shapes = len(x_shape) 
        return out.reshape(batch_size, *((1,) *  (n_shapes - 1))).to(t.device)

    # --- Forward Process ---
    def q_sample(self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_hat_t: Tensor = self.extract(self.sqrt_alphas_hat, t, x_0.shape) # Shape: [B, 1, 1, 1]
        sqrt_sigma_t: Tensor = self.extract(self.sigmas, t, x_0.shape)

        #  x_t = mean + std * noise -> mu + (sigma * z)
        x_t = (sqrt_alpha_hat_t * x_0) + (sqrt_sigma_t * noise)
        
        return x_t

    
    def forward(self, x_0: Tensor, x_cond: Tensor) -> Tensor:
        batch_size = x_0.shape[0]
        device = x_0.device

        # Random t time
        t: Tensor = torch.randint(0, self.timesteps, (batch_size, ), device=device).long() # Shape: [Batch] such as [5, 100, 999, ...]

        # Epsilon = N(0, I) Gaussian
        # 1000 path, 60 day, 14 assets , target feature = 1 (log return)
        # 776 , Batch size = 32
        # n_batch 20
        # 32, 60 day

        # 100,  , 14, 1 

        
        # Shape: [Batch, Window, Assets, Feature] 
        noise: Tensor = torch.randn_like(x_0) # pure noise

        x_t: Tensor = self.q_sample(x_0, t, noise=noise)

        predicted_noise: Tensor = self.model(x_t, t, x_cond) 

        # Compare between noise_predicted and pure noise 
        loss: Tensor = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    # --- Reverse Process ---
    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, t_index: int, x_cond: Tensor) -> Tensor:
        """
        x_t: preriod x_t (x has noise)
        t:   Tensor shape [Batch] such as [0,1,2,3,4,...,999] 
        t_index: index from t
        x_cond: condition 
        """
        # Predict noise
        predicted_noise = self.model(x_t, t, x_cond)

        # (eq.11) 1 / sqrt(alpha)
        coef_1 = self.extract(self.sqrt_recip_alpha, t, x_t.shape)
        # (eq.11) beta / sqrt(1 - alpha)
        coef_2 = self.extract(self.coef_noise, t, x_t.shape)
        # (eq.11) full equation
        model_mean = coef_1 * (x_t - (coef_2 * predicted_noise))

        if t_index > 0:
            noise = torch.randn_like(x_t)
            # x_{t-1} = mean + sqrt(Variance) * z 
            # N(mu, sigma^2)
            # Posterior_var = std^2 
            posterior_var_t = self.extract(self.posterior_variance, t, x_t.shape)
            return model_mean + (torch.sqrt(posterior_var_t) * noise)
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, x_cond: Tensor, output_channels: int) -> Tensor:
        B, W, A, _ = x_cond.shape
        device = x_cond.device
        
        # x with Pure Noise (x_T) 
        # x must have 1 Feature (Target Price) follow shape x_0
        # x = torch.randn((B, W, A, 1), device=device)
        x = torch.randn((B, W, output_channels), device=device)
        
        # (Timesteps - 1 to 0)
        # such as 999 -> 998 -> ... -> 0
        for i in reversed(range(0, self.timesteps)):
            # t as batch such as [999, 998, ...]
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            x = self.p_sample(x, t, i, x_cond)
            
        return x # x_0

    @torch.no_grad()
    def sample_inpaint(self, x_cond: Tensor, x_start: Tensor, mask: Tensor) -> Tensor:
        """
        Sampling with Inpainting (Replacement Method)
        Args:
            x_cond: Condition [B, W, A, F_cond]
            x_start: Ground Truth Data (The target we want to keep) [B, W, A, F]
            mask: Binary mask (1=Keep known, 0=Inpaint unknown) [B, W, A, F]
        """
        B, W, C = x_start.shape
        device = x_start.device
        
        # Pure Noise
        x_t = torch.randn_like(x_start)
        
        for i in reversed(range(0, self.timesteps)):
        # for i in  tqdm(reversed(range(0, self.timesteps)), desc="Inpainting", total=self.timesteps, leave=False):
            # Current t
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # let Model predicts next x_{t-1} (Predicted x_{t-1})
            x_t_minus_1_pred = self.p_sample(x_t, t, i, x_cond)
            
            # Known Part
            if i > 0:
                # Pure Noise 
                # x_{t-1} (Next step noise level)
                t_next = torch.full((B,), i - 1, device=device, dtype=torch.long)
                # q_sample noisy ground truth
                x_t_minus_1_known = self.q_sample(x_start, t_next) 
            else:
                # (t=0) x_0 so have no noise
                x_t_minus_1_known = x_start

            # Replacement with Mask
            # 1 = Known Part
            # 0 = Predicted Part
            x_t = (mask * x_t_minus_1_known) + ((1 - mask) * x_t_minus_1_pred)
            
        return x_t # Inpainted