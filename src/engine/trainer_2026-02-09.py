import torch
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional, Any, List, Dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from utils.paths import CHECKPOINTS_DIR, REPORTS_QS_DIR, REPORTS_SIM_DIR
from utils import inverse_scale, inverse_scale_pair, inspect, plot_to_base64, save_as_html 
from entities import Portfolio
from pypfopt import risk_models, expected_returns, plotting, EfficientFrontier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

    def expand_batch_with_n_sample(self, batch, n_samples: int):
        # [N Samples, W, A, F_cond] such as [10, 64, 14, 6]
        batch_expanded = {
            "x": torch.as_tensor(batch['x'].repeat(n_samples, 1, 1, 1), dtype=torch.float32).to(self.device),
            "x_cond": torch.as_tensor(batch['x_cond'].repeat(n_samples, 1, 1, 1), dtype=torch.float32).to(self.device)
        }
        return batch_expanded
    
    def transform_batch(self, batch):
        x_scaled, x_cond_scaled, _, _ = self.process_batch(batch)
        x, x_cond = inverse_scale_pair(x_scaled, x_cond_scaled, self.scaler)
        
        # logger.debug(f"x: {x.shape}\nx_cond: {x_cond.shape}\nx_scaled: {x_scaled.shape}\nx_cond_scaled: {x_cond_scaled.shape}")

        return x, x_cond, x_scaled, x_cond_scaled

    def transform_dates(self, dates: torch.Tensor) -> pd.DatetimeIndex:
        dates_np = dates.cpu().numpy()
        dates_pd = pd.to_datetime(dates_np.flatten(), unit='ns')
        return dates_pd
        
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
        full_simulation = inpainted_flat.view(B, W, A, F)

        # Extract Only Predicted Part
        if steps > 0:
            simulated_part = full_simulation[:, -steps:, :, :]
        else:
            simulated_part = torch.empty((B, 0, A, F), device=self.device)

        if inverse_scale:
            full_simulation, _ = inverse_scale_pair(full_simulation, cond_raw, self.scaler)
            simulated_part, _ = inverse_scale_pair(simulated_part, cond_raw[:, -steps:, :, :], self.scaler)
            x_real_raw, cond_raw = inverse_scale_pair(x_real_raw, cond_raw, self.scaler)
            
        # full simulate (groud truth and include simulate), groud truth, simutlation, condition (from groud truth) 
        return full_simulation, simulated_part, x_real_raw, cond_raw

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

    # autoregressive_simulate
    # Final output always be inversed scale
    def ar_simulate(self, x: np.ndarray, x_cond: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        B, W, A, F_target = x.shape
        period_curr = W - steps
        device = next(self.model.parameters()).device
    
        x_curr = torch.as_tensor(x[:, :period_curr, :, :], dtype=torch.float32).to(device)
        x_cond_curr = torch.as_tensor(x_cond[:, :period_curr, :, :], dtype=torch.float32).to(device)
    
        x_curr = x_curr.to(device)
        x_cond_curr = x_cond_curr.to(device)
        
        results = []
        print(f"Start Auto-regression: Initial W={period_curr}, Target Steps={steps}")
    
        pbar = tqdm(range(steps))
        for i in pbar:
            pbar.set_description(f"window size {x_curr.shape[1]}...")
            batch = {
                "x": x_curr,
                "x_cond": x_cond_curr
            }
    
            _, next_step_x, _, _ = self.simulate(batch, steps=1, inverse_scale=False)
            
            results.append(next_step_x)
            
            x_curr = torch.cat([x_curr, next_step_x], dim=1)
                
            idx = period_curr + i
                
            next_cond_slice_np = x_cond[:, idx : idx+1, :, :]
                
            next_cond_slice = torch.as_tensor(next_cond_slice_np, dtype=torch.float32).to(device)
    
            x_cond_curr = torch.cat([x_cond_curr, next_cond_slice], dim=1)
    
        scaled_sim_genai = torch.cat(results, dim=1).cpu().numpy()
        sim_genai, sim_genai_cond = inverse_scale_pair(scaled_sim_genai, x_cond_curr[:, -steps:, : ,:], self.scaler)
        
        scaled_full_sim_genai = x_curr.detach().cpu().numpy()
        full_sim_genai, full_sim_genai_cond = inverse_scale_pair(scaled_full_sim_genai, x_cond_curr, self.scaler)
        
        return full_sim_genai, sim_genai, full_sim_genai_cond, sim_genai_cond

    
    def armc_simulate(self, x: np.ndarray, x_cond: np.ndarray, steps: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Monte Carlo Simulation by expanding batch size.
        expand_shape = (n_samples, 1, 1, 1)
        
        x_expanded = np.tile(x, expand_shape)
        x_cond_expanded = np.tile(x_cond, expand_shape)

        return self.ar_simulate(x_expanded, x_cond_expanded, steps)

    def inspect_simulation(
        self,
        gt,
        gt_cond,
        sim_genai,
        sim_stats,
        sim_idx: Optional[int] = None,
        save_dir: Optional[str] = REPORTS_SIM_DIR,
        filename: Optional[str] = "sim",
        is_saved: bool = False
    ):
        # params: gt means ground truth
        logger.debug(f"\ngt: {gt.shape}\ngt_cond: {gt_cond.shape}\nsim_genai: {sim_genai.shape}\nsim_stats: {sim_stats.shape}")

        # inspect(gt, "Groud truth")
        # inspect(sim_genai, "Genai simulation")
        # inspect(sim_stats, "Statistic simulation")

        gt_df = pd.DataFrame(gt)
        sim_genai_df = pd.DataFrame(sim_genai)
        sim_stats_df = pd.DataFrame(sim_stats)

        # plotting.plot_covariance(risk_models.sample_cov(gt_df), plot_correlation=True)
        # plotting.plot_covariance(risk_models.sample_cov(sim_genai_df), plot_correlation=True)
        # plotting.plot_covariance(risk_models.sample_cov(sim_stats_df), plot_correlation=True)
        # plt.show()

        # Save as a Report
        
        plots = {}
        metrics = {
            "Ground Truth Mean Return": f"{gt_df.mean().mean():.4f}",
            "GenAI Mean Return": f"{sim_genai_df.mean().mean():.4f}",
            "Statistic Mean Return": f"{sim_stats_df.mean().mean():.4f}",
            "Ground Truth Volatility": f"{gt_df.std().mean():.4f}",
            "GenAI Volatility": f"{sim_genai_df.std().mean():.4f}",
            "Statistic Volatility": f"{sim_stats_df.std().mean():.4f}",
        }
        
        # 1. Ground Truth
        fig_gt, ax_gt = plt.subplots(figsize=(6, 5)) 
        corr_gt = risk_models.cov_to_corr(risk_models.sample_cov(gt_df))
        sns.heatmap(corr_gt, ax=ax_gt, cmap='coolwarm', center=0)
        ax_gt.set_title("Ground Truth Correlation")
        plots["Ground Truth Correlation"] = plot_to_base64(fig_gt)
        
        # 2. GenAI
        fig_gen, ax_gen = plt.subplots(figsize=(6, 5))
        corr_gen = risk_models.cov_to_corr(risk_models.sample_cov(sim_genai_df))
        sns.heatmap(corr_gen, ax=ax_gen, cmap='coolwarm', center=0)
        ax_gen.set_title("GenAI Correlation")
        plots["GenAI Correlation"] = plot_to_base64(fig_gen)

        # 3. Statistic
        fig_stat, ax_stat = plt.subplots(figsize=(6, 5))
        corr_stat = risk_models.cov_to_corr(risk_models.sample_cov(sim_stats_df))
        sns.heatmap(corr_stat, ax=ax_stat, cmap='coolwarm', center=0)
        ax_stat.set_title("Statistic Correlation")
        plots["Statistic Correlation"] = plot_to_base64(fig_stat)
        
        report = {
            "idx": sim_idx if sim_idx is not None else -1,
            "metrics": metrics,
            "plots": plots,
            "data": {
                "gt": gt.copy() if hasattr(gt, 'copy') else gt,
                "gt_cond": gt_cond.copy() if hasattr(gt_cond, 'copy') else gt_cond,
                "sim_genai": sim_genai.copy() if hasattr(sim_genai, 'copy') else sim_genai,
                "sim_stats": sim_stats.copy() if hasattr(sim_stats, 'copy') else sim_stats
            }
        }

        if is_saved:
            os.makedirs(save_dir, exist_ok=True)
            save_as_html(report, os.path.join(save_dir, f"{filename}_idx{sim_idx}.html"))

        return report
        
    def benchmark(
        self,
        gt: np.ndarray,
        sim_genai: np.ndarray,
        sim_stats: np.ndarray,
        dates: pd.DatetimeIndex,
        risk_free_rate: float = 0.02/252,
        is_calc_distribution: bool = False,
        save_dir: str = REPORTS_QS_DIR,
        filename: str = "benchmark"
    ):
        
        portfolio = Portfolio(risk_free_rate=risk_free_rate, weight_bounds=(0, 1), save_dir=save_dir)

        if is_calc_distribution:
            mu_genai, sigma_genai = portfolio.calc_distribution(sim_genai.tolist())
            mu_stats, sigma_stats = portfolio.calc_distribution(sim_stats.tolist())
        else:
            mu_genai, sigma_genai = portfolio.calc(sim_genai)
            mu_stats, sigma_stats = portfolio.calc(sim_stats)

        logger.debug(f"Mu_genai: {mu_genai.shape}, Sigma_genai: {sigma_genai.shape}")
        logger.debug(f"Mu_stats: {mu_stats.shape}, Sigma_stats: {sigma_stats.shape}")

        # portfolio.plot_ef(mu_genai, sigma_genai)
        # portfolio.plot_ef(mu_stats, sigma_stats)
        
        weights_genai = portfolio.optimize_weights(mu_genai, sigma_genai, risk_free_rate=risk_free_rate, scipy=False)
        weights_stats = portfolio.optimize_weights(mu_stats, sigma_stats, risk_free_rate=risk_free_rate, scipy=False)

        logger.debug(f"weights_genai: {weights_genai.shape}")
        logger.debug(f"weights_stats: {weights_stats.shape}")
        
        report = portfolio.back_test(weights=weights_genai, weights_benchmark=weights_stats, returns=gt, dates=dates, is_saved=True, filename=filename)
        
        return report

        
    def eval(self, batch, steps: int, n_samples: int, context_file:str =""):
        x, x_cond, x_scaled, x_cond_scaled = self.transform_batch(batch)
        dates = self.transform_dates(batch['dates'])
        
        # Inversed Scale!
        unscaled_batch = {
            "x": torch.as_tensor(x, dtype=torch.float32).to(self.device),
            "x_cond": torch.as_tensor(x_cond, dtype=torch.float32).to(self.device)
        }

        full_sim_gena, sim_genai, _, _ = self.armc_simulate(batch['x'], batch['x_cond'], steps, n_samples)
        sim_stats = self.gbm_simulate(unscaled_batch, steps=steps, n_samples=n_samples)

        sim_genai = sim_genai.squeeze(-1)
        sim_stats = sim_stats.squeeze(-1)
        gt = x[:, -steps:, :, :].squeeze(0).squeeze(-1)
        gt_cond = x_cond[:, -steps:, :, :].squeeze(0)
            
        # Overview
        date = f'sd{str(dates[0].strftime("%Y-%m-%d"))}_ed{str(dates[-1].strftime("%Y-%m-%d"))}'
        sim_dir = os.path.join(f"{context_file}", REPORTS_SIM_DIR, date)
        qs_dir = REPORTS_QS_DIR
        benchmark_filename = f"{context_file}benchmark_portfolio_{date}"

        print(f"Recording... overview simulations")
        self.inspect_simulation(
            gt=gt,
            gt_cond=gt_cond,
            sim_genai=sim_genai.mean(axis=0),
            sim_stats=sim_stats.mean(axis=0),
            sim_idx= None,
            save_dir=sim_dir,
            is_saved= True
        )
            
        self.benchmark(
            gt=gt,
            sim_genai=sim_genai,
            sim_stats=sim_stats,
            dates=dates[-steps:],
            is_calc_distribution=True,
            save_dir=qs_dir,
            filename=benchmark_filename
        )

        print(f"Recording... all simulations")
    
        all_reports = []
        save_path = os.path.join(sim_dir, "all_simulations.pkl")
            
        for i in range(len(sim_genai)):
            report = self.inspect_simulation(
                gt=gt,
                gt_cond=gt_cond,
                sim_genai=sim_genai[i, :, :],
                sim_stats=sim_stats[i, :, :],
                sim_idx= i,
                save_dir=sim_dir,
                is_saved= False
            )

            all_reports.append(report)
            
        with open(save_path, "wb") as f:
            pickle.dump(all_reports, f)
            
        print(f"Saved {len(all_reports)} reports to {save_path}")

    def load_simulation_report(self, file_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not Found: {file_path}")
    
        try:
            with open(file_path, "rb") as f:
                loaded_reports = pickle.load(f)
                
            if verbose:
                print(f"✅ Loaded {len(loaded_reports)} reports successfully.")
                if len(loaded_reports) > 0:
                    print("--- Example Metrics (First Item) ---")
                    print(loaded_reports[0].get("metrics", "No metrics key found"))
                    print("------------------------------------")
                    
            return loaded_reports
    
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            return []