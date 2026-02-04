import torch
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from utils.paths import CHECKPOINTS_DIR
from utils import inverse_scale, inverse_scale_pair, inspect
from entities import Portfolio
from pypfopt import risk_models, expected_returns, plotting, EfficientFrontier
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Require: model -> sample_inpaint()

class Engine:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader:   DataLoader,
        model:            nn.Module,
        scaler:           Union[StandardScaler, MinMaxScaler],
        optimizer:        Optimizer,
        device:           torch.device,
        criterion:        _Loss = nn.MSELoss(),
        save_dir:         str   = CHECKPOINTS_DIR,
        file_name:        str   = "ddpm_transformer"
    ):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader
        self.model            = model
        self.scaler           = scaler
        self.optimizer        = optimizer
        self.device           = device
        self.save_dir         = save_dir
        self.criterion        = criterion
        self.history          = {
                                  "train_losses": [],
                                  "val_losses":   []
                                }
        self.file_name = file_name
        self.best_val_loss = float('inf')
        
        os.makedirs(self.save_dir, exist_ok=True)

    def process_batch(self, batch):
        # Batch requires: [x, x_cond]
        x      = batch["x"].to(self.device).float()      # x:      [N, W, A, F]
        x_cond = batch["x_cond"].to(self.device).float() # x_cond: [N, W, A, F]
        
        batch_size, window_size, asset_size, feature_size = x.shape

        # Reshape (Flatten Assets)
        x_flat    = x.view(batch_size, window_size, -1)      # x:      [32, 64, 14, 1] -> [32, 64, 14*1] -> [32, 64, 14]
        x_cond_flat = x_cond.view(batch_size, window_size, -1) # x_cond: [32, 64, 14, 6] -> [32, 64, 14*6] -> [32, 64, 84]

        return x, x_cond, x_flat, x_cond_flat
    
    def transform_batch(self, batch):
        x_scaled, x_cond_scaled, _, _ = self.process_batch(batch)
        x, x_cond = inverse_scale_pair(x_scaled, x_cond_scaled, self.scaler)
        
        # logger.debug(f"x: {x.shape}\nx_cond: {x_cond.shape}\nx_scaled: {x_scaled.shape}\nx_cond_scaled: {x_cond_scaled.shape}")

        return x, x_cond, x_scaled, x_cond_scaled
    
    def train(self, epoch:int, epochs: int):
        self.model.train()

        train_loss_accum = 0
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            # Process Data
            _, _, x_flat, x_cond_flat = self.process_batch(batch)
            
            # Zero Grad (Remove old Grad)
            self.optimizer.zero_grad()
            
            # Forward
            loss = self.model(x_flat, x_cond_flat)
            
            # Backward (Calc new Grad)
            loss.backward()
            
            # Step (Update Weight)
            self.optimizer.step()
            
            # Logging
            train_loss_accum += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        return train_loss_accum / len(self.train_dataloader)

    def validate(self, epoch: int, epochs: int):
        self.model.eval()
        
        val_loss_accum = 0
        pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                _, _, x_flat, x_cond_flat = self.process_batch(batch)
                
                # Forward Pass Only
                val_loss = self.model(x_flat, x_cond_flat)
                val_loss_accum += val_loss.item()
                
        return val_loss_accum / len(self.val_dataloader)

    def fit(self, epochs: int):
        logger.info(f"Engine started Training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            
            # Train
            avg_train_loss = self.train(epoch, epochs)
            
            # Validate
            avg_val_loss = self.validate(epoch, epochs)
            
            # Record
            self.history["train_losses"].append(avg_train_loss)
            self.history["val_losses"].append(avg_val_loss)
            
            print(f"End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Save Checkpoints
            # Regular Save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{self.file_name}_e{epoch+1}.pt")
            
            # Save Best Model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(f"best_{self.file_name}_e{epoch + 1}.pt")
                print(f"New Best Model Saved (Val Loss: {self.best_val_loss:.6f})")
    
    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loaded model weights from {filename}")
        else:
            logger.warning(f"Checkpoint {filename} not found!")

    def simulate(self, batch, steps: int, inverse_scale: bool = False):
        x_real_raw, cond_raw, x_start_flat, cond_flat = self.process_batch(batch)

        B, W, A, F = x_real_raw.shape
        
        mask = torch.ones_like(x_real_raw)
        if steps > 0:
            mask[:, -steps:, :, :] = 0

        # Flatten Mask
        mask_flat = mask.view(B, W, -1)

        # Inpaint Execution
        self.model.eval()
        with torch.no_grad():
            inpainted_flat = self.model.sample_inpaint(
                x_cond=cond_flat,
                x_start=x_start_flat,
                mask=mask_flat
            )

        # Unflatten Output
        inpainted_final = inpainted_flat.view(B, W, A, F)

        # Extract Only Predicted Part
        if steps > 0:
            simulated_part = inpainted_final[:, -steps:, :, :]
        else:
            simulated_part = torch.empty((B, 0, A, F), device=self.device)

        if inverse_scale:
            inpainted_final, _ = inverse_scale_pair(inpainted_final, cond_raw, self.scaler)
            simulated_part, _ = inverse_scale_pair(simulated_part, cond_raw[:, -steps:, :, :], self.scaler)
            x_real_raw, cond_raw = inverse_scale_pair(x_real_raw, cond_raw, self.scaler)
            
        # full simulate (groud truth and include simulate), groud truth, simutlation, condition (from groud truth) 
        return inpainted_final, simulated_part, x_real_raw, cond_raw

    def mc_simulate(self, batch, steps: int, n_samples: int, inverse_scale: bool = False):
        # [Test Batch, W, A, F_cond] such as [1, 64, 14, 6]
        x_raw, cond_raw, _, _ = self.process_batch(batch)

        # [N Samples, W, A, F_cond] such as [10, 64, 14, 6]
        batch_expanded = {
            "x": x_raw.repeat(n_samples, 1, 1, 1),
            "x_cond": cond_raw.repeat(n_samples, 1, 1, 1)
        }

        full_sim, only_sim, x_raw, cond_raw = self.simulate(batch_expanded, steps, inverse_scale)

        return full_sim, only_sim, x_raw, cond_raw

    def gbm_simulate(self, batch, steps: int, n_samples: int):
        """Require: Unscaled batch only and must be log returns"""
        _, _, x_scaled_flat, _ = self.process_batch(batch)
        
        # x_scaled_flat: [B, W, A*F] -> [B, W, A] (Feature=1)
        B, W, _ = x_scaled_flat.shape
        x_scaled = x_scaled_flat.view(B, W, -1) 
        A = x_scaled.shape[-1]

        past_data = x_scaled[:, :-steps, :].cpu().numpy() # [B, Past_Steps, A]

        mu    = np.mean(past_data, axis=1) # [B, A]
        sigma = np.std(past_data, axis=1)  # [B, A]
        
        # Prepare for Broadcasting
        # [1, B, 1, A]
        mu    = mu[None, :, None, :]
        sigma = sigma[None, :, None, :]

        # Generate Random Returns (Normal Distribution)
        # Shape: [Samples, B, Steps, A]
        noise = np.random.normal(0, 1, (n_samples, B, steps, A))

        # Equation: r_t ~ N(mu, sigma)
        sim_returns = mu + sigma * noise

        sim_returns = sim_returns[:, 0, :, :]
        
        # Output Shape: [Samples, B, Steps, A, 1] (Fill Feature Dim = GenAI)
        return sim_returns[..., None].astype(np.float32)
        
    def eval(self, dataloader, steps: int, n_samples: int):
        for batch in tqdm(dataloader, desc="Evaluating"):
            x, x_cond, x_scaled, x_cond_scaled = self.transform_batch(batch)
            unscaled_batch = {
                "x": torch.tensor(x).to(self.device),
                "x_cond": torch.tensor(x_cond).to(self.device)
            }
            
            full_sim_genai, sim_genai, _, _ = self.mc_simulate(batch, steps, n_samples, inverse_scale=True)
            sim_stats = self.gbm_simulate(unscaled_batch, steps, n_samples)

            gt = torch.from_numpy(x[:, -steps:, :, :]).squeeze(0)
            gt_cond = torch.from_numpy(x_cond[:, -steps:, :, :]).squeeze(0)
            sim_stats = torch.from_numpy(sim_stats).squeeze(1)
            
            self.eval_simulations(gt, gt_cond, sim_genai, sim_stats)
            
    def eval_simulations(self, gt, gt_cond, sim_genai, sim_stats):
        
        logger.debug(f"\ngt: {gt.shape}\ngt_cond: {gt_cond.shape}\nsim_genai: {sim_genai.shape}\nsim_stats: {sim_stats.shape}")

        # Inspect boundary data
        rand_ind = np.random.randint(0, len(sim_genai))

        # Output Shape: [Steps, Assets, Feature (1 FT Exactly)]
        inspect(gt, "Groud truth")
        inspect(sim_genai[rand_ind], "Random 1 genai sim sample")
        inspect(sim_stats[rand_ind], "Random 1 stats sim sample")

        gt_df = pd.DataFrame(gt.squeeze(-1))
        sim_genai_df = pd.DataFrame(sim_genai[rand_ind].squeeze(-1))
        sim_stats_df = pd.DataFrame(sim_stats[rand_ind].squeeze(-1))

        plotting.plot_covariance(risk_models.sample_cov(gt_df), plot_correlation=True)
        plotting.plot_covariance(risk_models.sample_cov(sim_genai_df), plot_correlation=True)
        plotting.plot_covariance(risk_models.sample_cov(sim_stats_df), plot_correlation=True)
        plt.show()
        pass

    def benchmark(self, gt: np.ndarray, sim_genai: np.ndarray, sim_stats: np.ndarray, dates: pd.DatetimeIndex):
        risk_free_rate = 0.02 / 252 # Daily 
        
        portfolio = Portfolio(risk_free_rate=risk_free_rate, weight_bounds=(0, 1))

        mu_genai, sigma_genai = portfolio.calc(sim_genai)
        mu_stats, sigma_stats = portfolio.calc(sim_stats)

        # mu_genai, sigma_genai = portfolio.calc_distribution(sim_genai.tolist())
        # mu_stats, sigma_stats = portfolio.calc_distribution(sim_stats.tolist())

        logger.debug(f"Mu_genai: {mu_genai.shape}, Sigma_genai: {sigma_genai.shape}")
        logger.debug(f"Mu_stats: {mu_stats.shape}, Sigma_stats: {sigma_stats.shape}")

        portfolio.plot_ef(mu_genai, sigma_genai)
        portfolio.plot_ef(mu_stats, sigma_stats)
        
        weights_genai = portfolio.optimize_weights(mu_genai, sigma_genai, risk_free_rate=risk_free_rate, scipy=True)
        weights_stats = portfolio.optimize_weights(mu_stats, sigma_stats, risk_free_rate=risk_free_rate, scipy=True)

        logger.debug(f"weights_genai: {weights_genai.shape}")
        logger.debug(f"weights_stats: {weights_stats.shape}")
        
        report = portfolio.back_test(weights=weights_genai,weights_benchmark=weights_stats, returns=gt, dates=dates, is_saved=True,file_name="scipy_test")
        
        return report

        
    