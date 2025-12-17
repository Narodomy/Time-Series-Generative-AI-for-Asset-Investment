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

class Diffusion:
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
    
    def denoise_step(self, x_t, t, predicted_noise, noise=None):
        """ From Reverse Process step """
        """ 1d shape is [:, None, None] """
        alpha = self.alpha[t][:, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None]
        beta = self.beta[t][:, None, None]

        """ Calc Mean """
        noise_coeff = (1 - alpha) / torch.sqrt(1 - alpha_hat)
        mean = (1 / torch.sqrt(alpha)) * (x_t - noise_coeff * predicted_noise)

        if noise is None:
            return mean

        variance = torch.sqrt(beta) * noise
        return mean + variance
    
    def create_inpainting_mask(self, batch_size, known_length):
        """ Mask for inpainting (1 = known, 0 = forecast) """
        mask = torch.zeros((batch_size, self.length, self.channels)).to(self.device)
        mask[:, :known_length, :] = 1.0
        return mask

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
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = self.denoise_step(x_t=x, t=t, predicted_noise=predicted_noise, noise=noise)
        
        model.train()
        """ output shape(x) = [B, L, C] """
        return x 
        
    def sample_forecast(self, model, hist_series, horizon):
        model.eval()
        logging.info(f"🔮 Forecasting next {horizon} steps...")

        """ Check data size, obtained data + forecast = window_size """
        n_req_hist = self.length - horizon
        if len(hist_series) < n_req_hist:
            raise ValueError(f"History data too short! Need {req_hist} steps.")
        """ Get only necessary part of data """
        hist_part = hist_series[-n_req_hist:].values
        """ 
            Convert them to tensor, Create new mask 
            Shape [1, Hist, Channels]
        """
        x_known = torch.tensor(hist_part, dtype=torch.float32).unsqueeze(0).to(self.device) 
        mask = self.create_inpainting_mask(batch_size=1, known_length=hist_part)

        with torch.no_grad():
            """ Start with Radom Noise on linear """
            x = torch.randn((1, self.length, self.channels)).to(self.device)

            for i in tdqm(reversed(range(1, self.noise_steps)), postion=0):
                t = (torch.ones(1) * i).long().to(self.device)

                """ Denoise Step """
                predicted_noise = model(x, t)
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                """ Same Fomular with a sample function """
                x = self.denoise_step(x_t=x, t=t, predicted_noise=predicted_noise, noise=noise)

                """ In-painting Logic """
                if i > 1:
                    full_known_frame = torch.zeros_like(x)
                    full_known_frame[:, :n_req_hist, :] = x_known

                    """ Random Noise """
                    noise_known = torch.randn_like(full_known_frame)
                    """ Follow Forward Process Step of Diffusion """
                    known_noisy = torch.sqrt(alpha_hat) * full_known_frame + torch.sqrt(1 - alpha_hat) * noise_for_known
                    """ 
                        Reference: https://arxiv.org/pdf/2201.09865
                        RePaint: Inpainting using Denoising Diffusion Probabilistic Models 
                    """
                    x = mask * known_noisy + (1 - mask) * x

        model.train()

        full_seq = x.cpu().numpy()[0]
        """ Return (history part, forecast part) """
        return full_seq[:n_req_hist], full_seq[n_req_hist:]