import os
import logging
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from .strategies import DiffusionStrategy

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
        self,
        model,
        diffusion,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scaler=None,
        device="cuda",
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

        # loss fn
        self.criterion = nn.MSELoss()
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }
        
    def _process_batch(self, batch):
        """ Batch Shape: ["target", "context", "target_idx", "window_idx"] """
        if isinstance(batch, dict):
            if "target" in batch:
                x = batch["target"]
            else:
                raise KeyError(f"Batch dict missing 'target' key. Found: {batch.keys()}")
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        x = x.to(self.device).float()

        if x.ndim == 4:
            b, l, n, f = x.shape
            x = x.reshape(b, l, n*f)
            
        # logger.debug(f"X Shape: {x.shape}")
        return x

    def _create_mask(self, x, steps_to_predict):
        # 1=Known, 0=Predict
        mask = torch.ones_like(x).to(self.device)
        if steps_to_predict > 0:
            mask[:, -steps_to_predict:, :] = 0
        return mask
        
    def validate(self):
        self.model.eval()
        count = 0
        val_loss_sum = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                x = self._process_batch(batch)

                # Random t (Noise lv)
                t = torch.randint(0, self.diffusion.noise_steps, (x.shape[0],), device=self.device).long()

                # Noise (Forward)
                x_noisy, noise = self.diffusion.noise(x, t)

                # Let model predicts Noise
                predicted_noise = self.model(x_noisy, t)

                # Calc loss (Real noise vs Preidcted noise)
                loss = self.criterion(noise, predicted_noise)

                val_loss_sum += loss.item()
                count += 1
                
        self.model.train()
        return val_loss_sum / count if count > 0 else 0

    def fit(self, epochs, save_dir="checkpoints"):
        logger.info(f"Engine started Training for {epochs} epochs on {self.device}...")
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")

            epoch_loss_sum = 0
            count = 0

            for batch in pbar:
                # Prepare Data
                x = self._process_batch(batch)

                # Sample t (Random Timesteps) (0 to Batch size)
                t = torch.randint(0, self.diffusion.noise_steps, (x.shape[0],), device=self.device).long()

                # Forward Diffusion (Add Noise)
                x_noisy, noise = self.diffusion.noise(x, t)

                # Model Prediction
                predicted_noise = self.model(x_noisy, t)

                # Calc Loss
                loss = self.criterion(noise, predicted_noise)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Grad Clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                current_loss = loss.item()
                epoch_loss_sum += current_loss
                count += 1

                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = epoch_loss_sum / count
        
        # Validation
        avg_val_loss = 0
        val_msg = ""
        if self.val_dataloader is not None:
            avg_val_loss = self.validate()
            val_msg = f" | Val Loss: {avg_val_loss:.6f}"

        # Log History
        self.history["epoch"].append(epoch+1)
        self.history["train_loss"].append(avg_train_loss)
        self.history["val_loss"].append(avg_val_loss)
    
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}{val_msg}")

        # Save Logic
        self._save_checkpoint_logic(epoch, avg_train_loss, avg_val_loss, best_val_loss, save_dir)
        if avg_val_loss < best_val_loss and self.val_dataloader is not None:
            best_val_loss = avg_val_loss


    def simulate(self, x_input, steps_to_predict=1):
        """
            In-painting Simulation (Forecast)
            x_input: [Length, Channels] (Single sample) or [Batch, Length, Channels]
        """
        self.model.eval()
        x = torch.tensor(x_input) if not torch.is_tensor(x_input) else x_input.clone()

        # if batch is missing
        if x.ndim == 2:
            x = x.unsqueeze(0) # [1, L, C]

        x = x.to(self.device).float()

        # Flatten if 4D
        original_shape = None
        if x.ndim == 4:
            b, l, n, f = x.shape
            original_shape = (b, l, n, f)
            x = x.reshape(b, l, n*f)

        # Create Mask
        mask = self._create_mask(x, steps_to_predict)

        # Call DDPM inpainting
        logger.info(f"Simulating... (Masking last {steps_to_predict} steps)")
        x_filled = self.diffusion.sample_inpainting(self.model, x, mask)

        # Ensure known part is exact
        if steps_to_predict > 0:
            x_filled[:, :-steps_to_predict, :] = x[:, :-steps_to_predict, :]


        # Return original shape
        if original_shape is not None:
            x_filled = x_filled.reshape(original_shape)

        return x_filled.cpu().numpy()

    def _save_checkpoint_logic(self, epoch, train_loss, val_loss, best_val_loss, save_dir):
        # Save Best
        if self.val_dataloader is not None:
            if val_loss < best_val_loss:
                path = f"{save_dir}/best_model.pt"
                torch.save(self.model.state_dict(), path)
                logger.info(f"Saved Best Model at {path}")
        
        # Save Last
        path = f"{save_dir}/last_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
        }, path)