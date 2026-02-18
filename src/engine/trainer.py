import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Union
from utils.paths import CHECKPOINTS_DIR

logger = logging.getLogger("Engine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class Engine:
    def __init__(
        self,
        train_loader:     DataLoader,
        val_loader:       DataLoader,
        model:            nn.Module,
        optimizer:        optim.Optimizer,
        criterion:        nn.Module = nn.MSELoss(),
        scheduler:        torch.optim.lr_scheduler._LRScheduler = None,
        max_grad_norm:    float = 1.0,
        device:           torch.device = torch.device('cuda'),
        checkpoint_dir:   str    = CHECKPOINTS_DIR,
        checkpoint_filename: str = "ddpm_transformer.pt"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(f"Engine initialized on {device}")
        logger.info(f"Criterion: {criterion.__class__.__name__}")

    def _predict(self, x: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        # Helper Fn for calc loss
        pred_noise, noise = self.model(x, cond)
        loss = self.criterion(pred_noise, noise)
        return loss

    def train(self, epoch: int) -> float:
        self.model.train()
        
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", leave=True)

        for batch in pbar:
            if isinstance(batch, dict):
                x, cond = batch['x'], batch['cond']
            else:
                x, cond = batch

            x, cond = x.to(self.device), cond.to(self.device)
            # Optimization
            self.optimizer.zero_grad()
            loss = self._predict(x, cond)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Logging
            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        avg_loss = epoch_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Optional[float]:
        if self.val_loader is None:
            return None
            
        self.model.eval()
        val_loss = 0.0
        
        for batch in self.val_loader:
            if isinstance(batch, dict):
                x, cond = batch['x'], batch['cond']
            else:
                x, cond = batch
            
            x, cond = x.to(self.device), cond.to(self.device)
            
            # Forward pass only
            loss = self._predict(x, cond)
            val_loss += loss.item()
            
        avg_val_loss = val_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def fit(self, epochs: int, is_save_best: bool = True) -> None:
        logger.info(f"Starting training for {epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # 1. Train
            train_loss = self.train(epoch)
            
            # 2. Validate
            val_loss = self.validate(epoch)
            
            # 3. Scheduler Step
            if self.scheduler:
                # if it were ReduceLROnPlateau must return val_loss
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 4. Save Logic
            log_msg = f"Ep {epoch} | Train: {train_loss:.4f}"
            
            if val_loss is not None:
                log_msg += f" | Val: {val_loss:.4f}"
                
                # Save Best Model based on Validation Loss
                if is_save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_{self.checkpoint_filename}")
                    logger.info(f"New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.save_checkpoint(self.checkpoint_filename)

            logger.debug(log_msg)

        logger.info("Training Complete.")
        self.plot_history()

    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.history = ckpt['history']
            logger.info(f"Loaded checkpoint: {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
    
    def plot_history(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ------------------------------------------------------
    # Simulate
    # ------------------------------------------------------
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, cond: torch.Tensor):
        B, C_target, L, A = x.shape

        # # X, Cond shapes: [B, C, L, A]
        # mask = torch.ones_like(x).to(self.device)
        # mask[:, :, -steps:, :] = 0
        mask = self.model.create_lower_right_triangle_mask(x)

        self.model.eval()
        # # Inpainting
        x_inpainted = self.model.inpainting_sampler(x=x, cond=cond, mask=mask) # [B, C, L, A]

        sim_full = x_inpainted
        sim_only = x_inpainted[:, 1:, -1:, :]
        
        return sim_full, sim_only