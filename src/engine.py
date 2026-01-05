import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import save_model

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        diffusion,
        model,
        optimizer,
        num_epochs_to_save = 10,
        device="cuda"
    ):
        """ Variables Setup """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.diffusion = diffusion
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }
        
    @property
    def loss_fn(self):
        return torch.nn.MSELoss()
        
    def validate(self):
        self.model.eval()
        val_loss_sum = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
            x = x.to(self.device).float()
            t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise(x, t)
            predicted_noise = self.model(x_t, t)
            loss = self.loss_fn(noise, predicted_noise)
                
            val_loss_sum += loss.item()
            count += 1
            
        self.model.train()
        return val_loss_sum / count if count > 0 else 0
        
    def fit(self, epochs):
        logger.info(f"Engine started! Training for {epochs} epochs on {self.device}...")     

        best_val_loss = float('inf')
        
        loss_fn_name = type(self.loss_fn).__name__
        model_name = self.model.name if self.model.name is not None else "unknown"
        
        for epoch in range(epochs):             
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            
            epoch_loss_sum = 0
            count = 0
            
            for i, batch in enumerate(pbar):
                """ extract input from train dataloader, shape(x) = [B, L, C] """
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device).float()
                # x_raw = batch["x_raw"]
                """ fill noise into x (original input) """
                t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise(x, t)
                """ noise be predicted """
                predicted_noise = self.model(x_t, t)
                """ Loss function """
                loss = self.loss_fn(noise, predicted_noise)
                """ Optimizer steps """
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                epoch_loss_sum += current_loss
                count += 1
                """ Display Log """
                pbar.set_postfix({loss_fn_name: f"{current_loss:.4f}"})
            # Calc Train Avg. loss
            avg_train_loss = epoch_loss_sum / count

            # Calc Validation Avg. loss
            avg_val_loss = 0
            val_msg = ""
            if self.val_dataloader is not None:
                avg_val_loss = self.validate()
                val_msg = f" | Val Loss: {avg_val_loss:.6f}"

            # Display Log
            self.history["epoch"].append(epoch+1)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}{val_msg}")
            
            # Save Logic: Save every N epoch or save when new best val loss (New Best)
            if self.val_dataloader is not None:
                # if Val save when found the New Best Model (Best Practice)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint(epoch+1, avg_val_loss, "best_model.pt")
                    logging.info(f"New Best Model Found! (Val Loss: {best_val_loss:.6f})")
            else:
                # If it's not, just normal save
                if (epoch + 1) % save_interval == 0:
                    self._save_checkpoint(epoch+1, avg_train_loss, f"ckpt_ep{epoch+1}.pt")
                    
    def _save_checkpoint(self, epoch, loss, filename):
        # Wrapper function for save_model
        try:
             save_model(
                model=self.model, 
                optimizer=self.optimizer, 
                epoch=epoch, 
                loss=loss, 
                model_name=self.model.name if hasattr(self.model, 'name') else "diff_model", 
                loss_history=self.history["train_loss"], 
                epoch_history=self.history["epoch"]
            )
        except:
            pass