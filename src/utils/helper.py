import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils.paths import CHECKPOINTS_DIR
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf


def verify_scaling(t: torch.Tensor, dim: int= None):
    if dim is None:
        print(f"--- Global Stats ---")
        print(f"Max: {t.max().item():.4f}")
        print(f"Min: {t.min().item():.4f}")
        print(f"Mean: {t.mean().item():.4f}")
        print(f"Std: {t.std().item():.4f}")

    else:
        print(f"--- Stats along dim {dim} ---")
        max_val = t.max(dim=dim).values
        min_val = t.min(dim=dim).values
        mean_val = t.mean(dim=dim)
        std_val = t.std(dim=dim)

        print(f"Max: {max_val}") 
        print(f"Min: {min_val}")
        print(f"Mean: {mean_val}")
        print(f"Std: {std_val}")


def save_model(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    # 🟢 เพิ่ม Arguments ใหม่
    framework_name: str = "ddpm", 
    model_name: str = "unet",
    # 🟢 เปลี่ยน save_dir เป็น root เพื่อให้ยืดหยุ่น
    root_dir: str = CHECKPOINTS_DIR,
    loss_history: list = None,
    epoch_history: list = None
):
    """
    บันทึก Model, Optimizer ภายใต้โครงสร้าง: {root_dir}/{framework_name}/{model_name}/{epoch}_{date}.pt
    """
    
    # 1. กำหนด Path ของ Directory
    # เช่น "checkpoints/ddpm/unet"
    target_dir = os.path.join(root_dir)
    
    # 2. ตรวจสอบและสร้าง Directory (Recursive)
    os.makedirs(target_dir, exist_ok=True)
    
    # 3. กำหนดชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ชื่อไฟล์: 0050_20251126_212030.pt
    file_name = f"{framework_name.lower()}_{model_name.lower()}_{epoch:04d}_{timestamp}.pt" 
    save_path = os.path.join(target_dir, file_name)
    
    # 4. เตรียมข้อมูลที่จะบันทึก (Checkpoint Dictionary)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': timestamp,
        'framework': framework_name,
        'model_architecture': model_name
    }
    if loss_history is not None:
        checkpoint['loss_history'] = loss_history
    if epoch_history is not None:
        checkpoint['epoch_history'] = epoch_history
        
    # 5. บันทึก
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved successfully to: {save_path}")

def load_checkpoint(path: str):
    """
    Load Checkpoint from specified path and return all dictionaries
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
    checkpoint = torch.load(path)
    return checkpoint