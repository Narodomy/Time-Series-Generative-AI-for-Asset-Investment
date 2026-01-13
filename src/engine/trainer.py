import torch
import logging
from tqdm import tqdm
from utils import save_model
from .strategies import DiffusionStrategy

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, model, optimizer, diffusion, strategy: DiffusionStrategy):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.strategy = strategy

    def train_step(self, batch, device):
        self.optimizer.zero_grad()
        
        # 1. ให้ Strategy เตรียมข้อมูล (จะเป็น Joint หรือ Conditional ก็ช่างมัน)
        x_0 = self.strategy.prepare_batch(batch, device)
        
        # 2. Diffusion Process (Standard)
        t = self.diffusion.sample_timesteps(x_0.shape[0], device)
        noise = torch.randn_like(x_0)
        x_noisy = self.diffusion.q_sample(x_0, t, noise)
        
        # 3. ให้ Strategy คำนวณ Loss (จะเป็น MSE หรือ Stat ก็ช่างมัน)
        loss, log_dict = self.strategy.compute_loss(
            self.model, x_noisy, t, noise, 
            extra_info={'diffusion': self.diffusion} # ส่งตัวช่วยไปเผื่อใช้คำนวณ x0
        )
        
        # 4. Backward
        loss.backward()
        self.optimizer.step()
        
        return log_dict