import csv
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Annotated, Optional, Tuple

logger = logging.getLogger("Engine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class Engine:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        max_grad_norm: float = 1.0,
        clip_gradients: bool = True,
        device: torch.device = torch.device("cuda"),
        checkpoint_dir: str = "checkpoints",
        checkpoint_filename: str = "model.pt",
        seq_dims: Tuple[str, ...] = ("A",),
        feat_dims: Tuple[str, ...] = ("W", "D", "C"),
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.clip_gradients = clip_gradients
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_filename = checkpoint_filename
        self.seq_dims = seq_dims
        self.feat_dims = feat_dims

        # plots dir sits one level up from checkpoints/
        self.plots_dir = self.checkpoint_dir.parent / "plots"

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "grad_norm_mean": [],
            "grad_norm_max": [],
            "top_val_losses": [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # training_log.csv — write header once
        self._log_path = self.checkpoint_dir.parent / "training_log.csv"
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "grad_norm_mean",
                    "grad_norm_max",
                ]
            )

        logger.info(f"Engine initialised  device={device}")
        logger.info(f"Criterion : {criterion.__class__.__name__}")
        logger.info(f"Checkpoints → {self.checkpoint_dir}")
        logger.info(f"Plots      → {self.plots_dir}")

    # ──────────────────────────────────────────────────────────
    #  Core
    # ──────────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch["x"], batch["cond"]
        return batch[0], batch[1]  # TensorDataset → (x, cond, init_price, ohlcv)

    def _predict(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pred_noise, noise = self.model(x, cond)
        return self.criterion(pred_noise, noise)

    def train(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        grad_norms = []

        pbar = tqdm(
            self.train_loader, desc=f"Train Ep {epoch}", leave=True, mininterval=0.5
        )

        for batch in pbar:
            x, cond = self._unpack_batch(batch)
            x, cond = x.to(self.device), cond.to(self.device)

            self.optimizer.zero_grad()
            loss = self._predict(x, cond)
            loss.backward()

            if self.clip_gradients:
                raw_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            else:
                raw_norm = torch.norm(
                    torch.stack(
                        [
                            p.grad.norm()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    )
                )

            grad_norms.append(raw_norm.item())
            self.optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = epoch_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        self.history["grad_norm_mean"].append(float(np.mean(grad_norms)))
        self.history["grad_norm_max"].append(float(np.max(grad_norms)))

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Optional[float]:
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = 0.0
        batch_losses = []

        for batch in self.val_loader:
            x, cond = self._unpack_batch(batch)
            x, cond = x.to(self.device), cond.to(self.device)
            loss = self._predict(x, cond)
            val_loss += loss.item()
            batch_losses.append(loss.item())

        avg_val_loss = val_loss / len(self.val_loader)
        self.history["val_loss"].append(avg_val_loss)
        self.history["top_val_losses"].append(
            {
                "epoch": epoch,
                "top5": sorted(batch_losses, reverse=True)[:5],
            }
        )

        logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _append_log(self, epoch, train_loss, val_loss, lr, gnorm_mean, gnorm_max):
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}" if val_loss is not None else "",
                    f"{lr:.8f}",
                    f"{gnorm_mean:.4f}",
                    f"{gnorm_max:.4f}",
                ]
            )

    # ──────────────────────────────────────────────────────────
    #  Fit
    # ──────────────────────────────────────────────────────────

    def fit(self, epochs: int, is_save_best: bool = True, save_every: int = 10) -> None:
        logger.info(f"Training starts — {epochs} epochs")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):

            train_loss = self.train(epoch)
            val_loss = self.validate(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Scheduler step
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # CSV log — append every epoch so crash won't lose history
            self._append_log(
                epoch,
                train_loss,
                val_loss,
                current_lr,
                self.history["grad_norm_mean"][-1],
                self.history["grad_norm_max"][-1],
            )

            # Save best
            if val_loss is not None and is_save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                logger.info(f"  ↓ best model saved  (val={val_loss:.4f})")

            # Periodic checkpoint
            if save_every > 0 and epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt")

            logger.debug(
                f"Ep {epoch} | train={train_loss:.4f}"
                + (f" | val={val_loss:.4f}" if val_loss else "")
                + f" | lr={current_lr:.2e}"
            )

        logger.info("Training complete.")
        self.save_all_plots()

    # ──────────────────────────────────────────────────────────
    #  Checkpoint
    # ──────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.history = ckpt["history"]
            logger.info(f"Loaded checkpoint: {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")

    # ──────────────────────────────────────────────────────────
    #  Plots
    # ──────────────────────────────────────────────────────────

    def save_all_plots(self) -> None:
        self._plot_loss_curve()
        self._plot_lr_and_gradients()
        self._plot_val_worst_batches()
        self._plot_training_summary()
        logger.info(f"All plots saved → {self.plots_dir}")

    def _plot_loss_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        ax.plot(epochs, self.history["train_loss"], label="Train", linewidth=1.5)
        if self.history["val_loss"]:
            ax.plot(
                epochs,
                self.history["val_loss"],
                label="Val",
                linestyle="--",
                linewidth=1.5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_lr_and_gradients(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        epochs = range(1, len(self.history["lr"]) + 1)

        # top — LR
        ax1.plot(epochs, self.history["lr"], color="steelblue", linewidth=1.5)
        ax1.set_ylabel("Learning Rate")
        ax1.set_title("Learning Rate Schedule")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # bottom — grad norms
        ax2.plot(
            epochs,
            self.history["grad_norm_mean"],
            label="Mean",
            color="darkorange",
            linewidth=1.5,
        )
        ax2.plot(
            epochs,
            self.history["grad_norm_max"],
            label="Max",
            color="tomato",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Grad Norm")
        ax2.set_title("Gradient Norms")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "lr_and_gradients.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def _plot_val_worst_batches(self) -> None:
        if not self.history["top_val_losses"]:
            return

        records = self.history["top_val_losses"]
        epochs = [r["epoch"] for r in records]
        top1 = [r["top5"][0] for r in records]
        top5_mean = [np.mean(r["top5"]) for r in records]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, top1, label="Worst batch", color="tomato", linewidth=1.5)
        ax.plot(
            epochs,
            top5_mean,
            label="Top-5 mean",
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )

        if self.history["val_loss"]:
            ax.plot(
                range(1, len(self.history["val_loss"]) + 1),
                self.history["val_loss"],
                label="Avg val loss",
                color="steelblue",
                linestyle=":",
                linewidth=1,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Validation — Worst Batches per Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "val_worst_batches.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def _plot_training_summary(self) -> None:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # ── top-left: loss ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history["train_loss"], label="Train", linewidth=1.5)
        if self.history["val_loss"]:
            ax1.plot(
                epochs,
                self.history["val_loss"],
                label="Val",
                linestyle="--",
                linewidth=1.5,
            )
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── top-right: lr ────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history["lr"], color="steelblue", linewidth=1.5)
        ax2.set_title("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # ── bottom-left: grad norms ──────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            epochs,
            self.history["grad_norm_mean"],
            label="Mean",
            color="darkorange",
            linewidth=1.5,
        )
        ax3.plot(
            epochs,
            self.history["grad_norm_max"],
            label="Max",
            color="tomato",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax3.set_title("Gradient Norms")
        ax3.set_xlabel("Epoch")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ── bottom-right: worst batches ──────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        if self.history["top_val_losses"]:
            records = self.history["top_val_losses"]
            ep_list = [r["epoch"] for r in records]
            top1 = [r["top5"][0] for r in records]
            top5_mean = [np.mean(r["top5"]) for r in records]
            ax4.plot(ep_list, top1, label="Worst batch", color="tomato", linewidth=1.5)
            ax4.plot(
                ep_list, top5_mean, label="Top-5 mean", color="orange", linestyle="--"
            )
        ax4.set_title("Val Worst Batches")
        ax4.set_xlabel("Epoch")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
        fig.savefig(
            self.plots_dir / "training_summary.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # ──────────────────────────────────────────────────────────
    #  Simulate / Trajectory
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def simulate(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Annotated[torch.Tensor, ("B", "A", "W", "C")]:
        B, A, W, C = x.shape
        self.model.eval()
        return self.model.sample(x_cond=cond, output_shape=(B, A, W, C))

    @torch.no_grad()
    def plot_denoising_trajectory(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        asset_idx: int = 0,
        channel_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        self.model.eval()
        trajectory = self.model.sample_trajectory(x, cond)
        n_steps = len(trajectory)
        n_panels = min(10, n_steps)
        indices = np.linspace(0, n_steps - 1, n_panels, dtype=int)

        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3, 4))
        if n_panels == 1:
            axes = [axes]

        for plot_i, step_i in enumerate(indices):
            frame = trajectory[step_i][0, asset_idx, :, channel_idx].cpu().numpy()
            ax = axes[plot_i]
            ax.plot(frame)
            ax.set_title(f"t={n_steps - 1 - step_i}")
            ax.set_xlabel("Token")
            ax.tick_params(left=False, labelleft=False)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(f"Asset {asset_idx} | Ch {channel_idx}")
        fig.suptitle(r"Denoising Trajectory $x_T \rightarrow x_0$", fontsize=13)
        plt.tight_layout()

        out = save_path or str(self.plots_dir / "denoising_trajectory.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Trajectory saved → {out}")
