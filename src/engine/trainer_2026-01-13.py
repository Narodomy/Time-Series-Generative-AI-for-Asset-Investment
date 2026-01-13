import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
import os

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
        device=torch.device("cuda"),
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
        x = batch [0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device).float()

        if x.ndim == 4:
            b, l, n, f = x.shape
            x = x.reshape(b, l, n*f)

        return x
        
        
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

    # def simulate(self, n_samples=64, steps_to_predict=1 ,test_dataloader=None):
    #     # X real = Ground Truth, X Filled = In-painted
    #     loader = test_dataloader if test_dataloader is not None else self.test_dataloader
    #     if loader is None:
    #         raise ValueError("Test Dataloader is not defined.")
            
    #     logger.info(f"Starting Simulate (Masking last {steps_to_predict} steps) on {n_samples} samples...")
        
    #     self.model.eval()
        
    #     results_true = []
    #     results_pred = []
    #     count = 0

    #     # Pull Test set
    #     with torch.no_grad():
    #         for batch in loader:
    #             if count >= n_samples:
    #                 break

    #             # Prepare Data
    #             if isinstance(batch, (list, tuple)):
    #                 x_real = batch[0]
    #             else:
    #                 x_real = batch
                    
    #             current_batch_size = x_real.shape[0]
    #             if count + current_batch_size > n_samples:
    #                 needed = n_samples - count
    #                 x_real = x_real[:needed]

    #             x_real = x_real.to(self.device).float() # [Batch, Window_Size, Features]
    #             batch_size, seq_len, features = x_real.shape
                
    #             # Create Mask
    #             # Logic: 1 = Known (Context), 0 = Unknown (To Predict)
    #             mask = torch.ones_like(x_real).to(self.device)

    #             # For examples: seq_len=64, steps=1 -> Close at 63 (index -1)
    #             #               seq_len=64, steps=5 -> Close at 59-63 (index -5 to last one)
    #             mask[:, -steps_to_predict:, :] = 0

    #             # In-painting
    #             x_filled = self.diffusion.sample_inpainting(self.model, x_real, mask)
        
    #             # Ground Truth (Normalized)
    #             if steps_to_predict > 0:
    #                 x_filled[:, :-steps_to_predict, :] = x_real[:, :-steps_to_predict, :]
    #             # Forcasted (Normalized)
    #             results_true.append(x_real.cpu().numpy())
    #             results_pred.append(x_filled.cpu().numpy())
                
    #             count += x_real.shape[0]

    #     # Shape output: [n_samples, steps_to_predict, features]
    #     final_true = np.concatenate(results_true, axis=0)
    #     final_pred = np.concatenate(results_pred, axis=0)
        
    #     logger.info(f"Simulate finished. Shape: {final_pred.shape}")
        
    #     return final_true, final_pred

    def simulate(self, x, steps_to_predict=1):
        """
        Pure Function: รับ Tensor เข้ามาทำ In-painting แล้วจบ
        ไม่สนใจ Dataloader หรือ Loop ภายนอก
        """
        self.model.eval()
        
        # 1. จัดทรง (ใช้ Adapter ตัวเก่งของเรา)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(self.device).float()
        
        # เรียก Adapter (สมมติว่าพี่มี Class TensorAdapter จากรอบที่แล้ว)
        # ถ้ายังไม่มี ให้ใช้ x_std = x.reshape(...) ธรรมดาไปก่อน
        x_std, meta = TensorAdapter.to_model_input(x)
        
        # 2. สร้าง Mask
        mask = torch.ones_like(x_std).to(self.device)
        if steps_to_predict > 0:
            mask[:, -steps_to_predict:, :] = 0  # ปิดท้าย
            
        # 3. เข้าเตาอบ (Diffusion)
        with torch.no_grad():
            x_filled = self.diffusion.sample_inpainting(self.model, x_std, mask)
            
            # 4. Fix Context (เอาของจริงแปะทับส่วนที่ไม่โดนบัง)
            if steps_to_predict > 0:
                x_filled[:, :-steps_to_predict, :] = x_std[:, :-steps_to_predict, :]
                
        # 5. คืนร่างเดิม
        x_out = TensorAdapter.restore_output(x_filled, meta)
        
        return x_out.cpu().numpy()
    
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