import torch
import logging
import math
import torch.nn.functional as F

from tqdm import tqdm

""" 
    Note!
    shape(x) = [B, L, C]
    B means Batch size
    L means Length (Window size)
    C means Channels (Features)
"""
logger = logging.getLogger(__name__)

class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="cosine",
        device=torch.device("cuda")
    ):
        # Variable Setup
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Prepare Schedule
        self.betas = self.prepare_noise_schedule(schedule).to(device)
        
        # alpha_t = 1 - beta_t
        self.alphas = 1. - self.betas 

        # q(x_t | x_0) call that Cumulative Product
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # alpha_bar_{t-1} to calc Posterior Variance
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-Calced Constants
        # DDPM equation: q(x_t | x_0) = N(x_t; sqrt(alpha_bar)*x_0, (1-alpha_bar)I)

        # Coefficient forward x_0 (Mean part)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # Coefficient forward epsilon/noise (Variance part)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 1 / sqrt(alpha_t) -> mutiply x_{t-1} (Denoise step/sampling)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior q(x_{t-1} | x_t, x_0)
        # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def prepare_noise_schedule(self, schedule):
        logger.debug(f"Diffusion is using {schedule} schedule.")
        
        if schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif schedule == "cosine":
            return self.cosine_variance_schedule(self.noise_steps) # Improved DDPM (Better for low noise levels)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
    def cosine_variance_schedule(self, timesteps, s=0.008):
        """ Standard Cosine Schedule from OpenAI/Nichol paper """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def extract(self, a, t, x_shape):
        """ 
            Pull a at t (time step) and reshape it to compatible with x
            Can use both [B, T, F] and [B, T, N, F]
            Arg:
                t = Noise Level
                a = alpha
            Output shape be like: [Batch num, 1, 1 ,1] or [Batch num, 1, 1]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1)))

    # Forward Process
    def noise(self, x, t):
        """
            Forward Diffusion Process
            Reference: DDPM Eq. 4
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        noise = torch.randn_like(x)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        x_noisy = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return x_noisy, noise

    # Reverse Process
    def denoise_step(self, model, x, t, t_index):
        """
            Reverse Diffusion Process (Sampling)
            Reference: DDPM Eq. 11 (Algorithm 2)
            mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
        """
        # Coefficients
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, x.shape)


        # Predict Noise (epsilon_theta)
        predicted_noise = model(x, t)

        # Calculate Mean (mu_theta) from Eq. 11
        model_mean = sqrt_recip_alpha_t * (
            x - (betas_t * predicted_noise) / sqrt_one_minus_alpha_bar_t
        )

        # Add Variance (sigma_t * z) if t > 0
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            
            # Using posterior_variance (beta_tilde) instead the beta that follows Improved DDPM
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def sample(self, model, shape):
        logger.info(f"Sampling shape {shape}...")
        model.eval()
        with torch.no_grad():
            x = torch.randn(shape).to(self.device)
            
            for i in tqdm(reversed(range(0, self.noise_steps)), desc="Sampling", total=self.noise_steps):
                t = (torch.ones(shape[0]) * i).long().to(self.device)
                x = self.denoise_step(model, x, t, i)
                
        model.train()
        return x