import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import save_model

class Engine:
    def __init__(
        self,
        dataloader,
        diffusion,
        model,
        optimizer,
        loss_fn=None,
        num_epochs_to_save = 10,
        device="cuda"
    ):
        """ Variables Setup """
        self.dataloader = dataloader
        self.diffusion = diffusion
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.history = {
            "epoch": [],
            "loss": []
        }
        
    def fit(self, epochs):
        logging.info(f"Engine started! Training for {epochs} epochs on {self.device}...")     
        loss_fn_name = type(self.loss_fn).__name__
        model_name = self.model.name if self.model.name is not None else "unknown"
        for epoch in range(epochs):            
            self.model.train()
            
            pbar = tqdm(self.dataloader)
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            
            epoch_loss_sum = 0
            count = 0
            
            for i, batch in enumerate(pbar):
                """ extract input from dataloader, shape(x) = [B, L, C] """
                x = batch["x"].to(self.device)
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
            """ Calc Loss Average """
            avg_loss = epoch_loss_sum / count
            """ Record history """
            self.history["epoch"].append(epoch)
            self.history["loss"].append(avg_loss)
            if epoch % num_epochs_to_save == 0 or epoch == epochs:
                save_model(model=self.model, optimizer=self.optimizer, epoch=epoch, loss=current_loss, model_name=model_name, loss_history=self.history["loss"], epoch_history=self.history["epoch"])
            """ Display Log """
            logging.info(f"Epoch {epoch} | Avg {loss_fn_name}: {avg_loss:.6f}")

    def plot_loss(self):
            plt.figure(figsize=(10, 5))
            plt.plot(self.history["epoch"], self.history["loss"], label="Training Loss")
            plt.title(f"Training Loss ({type(self.loss_fn).__name__}) Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()