import torch
import logging
from tqdm import tqdm

""" 
    Note!
    shape(x) = [B, L, C]
    B means Batch size
    L means Length (Window size)
    C means Channels (Features)
"""

class GaussianDiffusion:
    def __init__(
        self, 
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        length=60, 
        channels=2,
        device="cuda"
    ):
        """ Variables Setup """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.length = length
        self.channels = channels
        self.device = device
        
        """ Calculation """ 
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) 

    def noise(self, x, t):
        """ shape(x) = [B, L, C]  """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        E = torch.randn_like(x)
        return (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha_hat * E), E
    
    def sample_timesteps(self, n):
        """ Generate noises from 1 to noise steps (if noise steps = 1000) at size = 5 
            means new 5 values into array (from 1 to 1000) 
            such as n = 5, [1,100,200, 500, 1000] """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} new time series windows....")
        model.eval()
        with torch.no_grad():
            """ Start random noises, shape(x) = [B, L, C] """
            x = torch.randn((n, self.length, self.channels)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                
                """ 1d shape is [:, None, None] """
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        """ output shape(x) = [B, L, C] """
        return x 