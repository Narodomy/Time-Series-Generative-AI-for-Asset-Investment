import torch
import numpy as np
from .visualization import visualize_all

def get_real_batch(dataloader, n_samples):
    """Prepared real data"""
    real_batch = next(iter(dataloader))
    return real_batch["x"].cpu().numpy()[:n_samples]

def get_fake_batch(diffusion, model, n_samples):
    """Order to model generate data"""
    # model.eval() # ปกติใน diffusion.sample มีเรียก eval ให้อยู่แล้ว แต่ใส่กันเหนียวได้
    with torch.no_grad():
        x_fake = diffusion.sample(model, n=n_samples)
    return x_fake.cpu().numpy()

def evaluate_model(diffusion, model, dataloader, device, n_samples=100):
    """
    Main Function: Orchestrator
    1. Prepare Data
    2. Visual Evaluation
    """
    print(f"📊 Starting Evaluation (Samples: {n_samples})...")
    
    # 1. Prepare Data
    x_real = get_real_batch(dataloader, n_samples)
    x_fake = get_fake_batch(diffusion, model, n_samples)
    
    # 2. Visualize
    visualize_all(x_real, x_fake)
    
    print("✅ Evaluation Complete.")