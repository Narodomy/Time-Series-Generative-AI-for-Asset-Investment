import os
import json
import logging
import torch
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from dataclasses import asdict
from datetime import datetime
from utils.paths import EXPERIMENTS_DIR
import optuna.visualization.matplotlib as optuna_plt
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
)


class ExperimentManager:
    def __init__(
        self, config, exp_name: str = "ddpm_run", base_dir: str = EXPERIMENTS_DIR
    ):
        """
        รับ config เข้ามาเลย เพื่อจัดการชื่อโฟลเดอร์และเซฟค่าอัตโนมัติ
        """
        self.config = config

        # 1. เช็คโหมด: ถ้า skip_optuna เป็น True ให้ต่อท้ายชื่อว่า _baseline ถ้า False ให้เป็น _optuna
        mode_suffix = "baseline" if config.skip_optuna else "optuna"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{timestamp}_{exp_name}_{mode_suffix}"
        self.exp_dir = os.path.join(base_dir, self.exp_id)

        # 2. สร้างโครงสร้างโฟลเดอร์
        self.dirs = {
            "checkpoints": os.path.join(self.exp_dir, "checkpoints"),
            "plots": os.path.join(self.exp_dir, "plots"),
            "reports": os.path.join(self.exp_dir, "reports"),
            "logs": os.path.join(self.exp_dir, "logs"),
        }

        # otuna results directory (only if not skipping)
        if not config.skip_optuna:
            self.dirs["optuna"] = os.path.join(self.exp_dir, "optuna")

        # eval results directory
        if not config.skip_evaluation:
            self.dirs["eval"] = os.path.join(self.exp_dir, "eval")

        self._create_directories()
        self._setup_logger()

        # 3. เซฟ Config ทันทีตอนเริ่มรัน
        self._save_config()

    def _create_directories(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def _setup_logger(self):
        log_file = os.path.join(self.dirs["logs"], "training.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ExpManager")

    def _save_config(self):
        """บันทึกพารามิเตอร์เก็บไว้กันลืม (จัดการพวก object ที่เซฟลง JSON ตรงๆ ไม่ได้)"""
        config_dict = asdict(self.config)

        # แก้ปัญหา torch.device เซฟลง JSON ไม่ได้
        if "device" in config_dict:
            config_dict["device"] = str(config_dict["device"])

        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

    def get_path(self, folder_type: str, filename: str) -> str:
        if folder_type not in self.dirs:
            raise ValueError(f"Unknown folder type: {folder_type}")
        return os.path.join(self.dirs[folder_type], filename)

    def save_plot(self, fig: plt.Figure, filename: str):
        save_path = self.get_path("plots", filename)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def _save_optuna_html_plots(self, study: optuna.Study):
        """Save interactive Optuna plots as HTML files using Plotly backend."""
        opt_dir = self.dirs["optuna"]

        html_plots = {
            "optimization_history.html": plot_optimization_history,
            "parallel_coordinate.html": plot_parallel_coordinate,
            "param_importances.html": plot_param_importances,
            "slice_plot.html": plot_slice,
        }

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        for filename, plot_fn in html_plots.items():
            # param_importances and slice_plot need at least 2 completed trials
            if (
                filename in ("param_importances.html", "slice_plot.html")
                and len(completed_trials) < 2
            ):
                self.logger.warning(
                    f"Not enough completed trials to generate {filename} — skipping."
                )
                continue
            try:
                fig = plot_fn(study)
                fig.write_html(os.path.join(opt_dir, filename))
            except Exception as e:
                self.logger.warning(f"Failed to generate {filename}: {e}")

        self.logger.info("Interactive HTML plots saved to optuna/")

    def _plot_study_results(self, study: optuna.Study):
        """Plot four key Optuna diagnostics and save as a single PNG."""
        trials_df = study.trials_dataframe()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Optimization history: loss trajectory across all trials ---
        ax = axes[0, 0]
        ax.plot(trials_df["number"], trials_df["value"], "o-", alpha=0.6)
        ax.axhline(
            study.best_value,
            color="r",
            linestyle="--",
            label=f"Best: {study.best_value:.6f}",
        )
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Timesteps impact: loss distribution per timestep setting ---
        ax = axes[0, 1]
        sorted_ts = sorted(trials_df["params_timesteps"].unique())
        ts_data = [
            trials_df[trials_df["params_timesteps"] == ts]["value"].tolist()
            for ts in sorted_ts
        ]
        ax.boxplot(ts_data, labels=sorted_ts)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Impact of Timesteps")
        ax.grid(True, alpha=0.3)

        # --- Learning rate impact: scatter on log scale ---
        ax = axes[1, 0]
        ax.scatter(trials_df["params_lr"], trials_df["value"], alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Learning Rate Impact")
        ax.grid(True, alpha=0.3)

        # --- Model dimension impact: loss distribution per d_model value ---
        ax = axes[1, 1]
        sorted_dm = sorted(trials_df["params_d_model"].unique())
        dm_data = [
            trials_df[trials_df["params_d_model"] == dm]["value"].tolist()
            for dm in sorted_dm
        ]
        ax.boxplot(dm_data, labels=sorted_dm)
        ax.set_xlabel("d_model")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Impact of Model Dimension")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, "study_results.png")
        self.logger.info("Study result plots saved to plots/study_results.png")

    def save_optuna_study(self, study: optuna.Study):
        """Save Optuna study artifacts: best params, trial history, and diagnostic plots."""
        if self.config.skip_optuna:
            self.logger.warning("skip_optuna=True — skipping Optuna save.")
            return

        opt_dir = self.dirs["optuna"]

        # Save best hyperparameters as JSON
        with open(os.path.join(opt_dir, "best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)

        # Save full trial history as CSV
        study.trials_dataframe().to_csv(
            os.path.join(opt_dir, "all_trials.csv"), index=False
        )

        # Generate interactive HTML plots (Plotly) and static PNG (Matplotlib)
        try:
            self._save_optuna_html_plots(study)
            self._plot_study_results(study)
        except Exception as e:
            self.logger.error(f"Failed to save study plots: {e}")

    def save_model_summary(self, model: torch.nn.Module):
        """Save model architecture details: layer breakdown, parameter counts, and config."""

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Build per-layer breakdown
        layer_summary = [
            {
                "layer": name,
                "shape": list(param.shape),
                "params": param.numel(),
                "trainable": param.requires_grad,
            }
            for name, param in model.named_parameters()
        ]

        summary = {
            "model_class": model.__class__.__name__,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params,
            "layers": layer_summary,
        }

        path = os.path.join(self.dirs["reports"], "model_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(
            f"Model summary saved — {trainable_params:,} trainable params "
            f"out of {total_params:,} total."
        )

    def save_training_results(self, engine):
        """Save full training artifacts: loss curves (PNG), history (CSV), and run report (JSON)."""

        history = engine.history  # expects dict with "train_loss" and "val_loss" lists

        # --- LR + Gradient flow subplot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(history["lr"])
        axes[0].set_title("Learning Rate Schedule")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("LR")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["grad_norm_mean"], label="Mean")
        axes[1].plot(history["grad_norm_max"], label="Max", alpha=0.5)
        axes[1].set_title("Gradient Norm per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Grad Norm")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, "lr_and_gradients.png")

        # --- Loss curve plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax.plot(epochs, history["train_loss"], label="Train Loss")
        ax.plot(epochs, history["val_loss"], label="Val Loss")

        best_epoch = int(pd.Series(history["val_loss"]).idxmin()) + 1
        best_val = min(history["val_loss"])
        ax.axvline(
            best_epoch,
            color="r",
            linestyle="--",
            alpha=0.6,
            label=f"Best epoch {best_epoch} ({best_val:.6f})",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_plot(fig, "loss_curve.png")

        # --- Loss history CSV ---
        history_df = pd.DataFrame(
            {
                "epoch": list(epochs),
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
            }
        )
        history_df.to_csv(
            os.path.join(self.dirs["reports"], "loss_history.csv"), index=False
        )

        # --- Run report JSON (training summary + config snapshot) ---
        report = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "total_epochs": len(history["train_loss"]),
            "config": {
                "lr": self.config.lr,
                "epochs": self.config.epochs,
                "d_model": self.config.d_model,
                "noise_steps": self.config.noise_steps,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "dropout": self.config.dropout,
                "weight_decay": self.config.weight_decay,
                "beta_start": self.config.beta_start,
                "beta_end": self.config.beta_end,
            },
        }

        path = os.path.join(self.dirs["reports"], "run_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=4)

        self.logger.info(
            f"Training results saved — best val loss {best_val:.6f} at epoch {best_epoch}."
        )

    # def save_evaluation_results(
    #     self,
    #     all_metrics: list,
    #     all_gen_stats: list,
    #     all_weights: list,
    # ) -> None:
    #     if self.config.skip_evaluation:
    #         self.logger.warning("skip_evaluation=True — skipping evaluation save.")
    #         return

    #     eval_dir = self.dirs["eval"]

    #     # 1. generation stats ทุก sample รวมกัน
    #     df_gen_all = pd.concat(
    #         [df.assign(sample_idx=i) for i, df in enumerate(all_gen_stats)],
    #         ignore_index=True,
    #     )
    #     df_gen_all.to_csv(
    #         os.path.join(eval_dir, "generation_stats_all.csv"), index=False
    #     )

    #     # 2. QuantStats metrics ทุก sample side-by-side
    #     metrics_combined = pd.concat(
    #         [df.add_suffix(f"_s{i}") for i, df in enumerate(all_metrics)],
    #         axis=1,
    #     )
    #     metrics_combined.to_csv(os.path.join(eval_dir, "backtest_metrics_all.csv"))

    #     # 3. summary JSON
    #     sharpe_vals, cagr_vals, maxdd_vals = [], [], []
    #     for df in all_metrics:
    #         sharpe_row = df[df.index.str.contains("sharpe", case=False)]
    #         cagr_row = df[df.index.str.contains("cagr", case=False)]
    #         maxdd_row = df[df.index.str.contains("max.*draw", case=False)]
    #         if not sharpe_row.empty:
    #             sharpe_vals.append(float(sharpe_row.iloc[0, 0]))
    #         if not cagr_row.empty:
    #             cagr_vals.append(float(cagr_row.iloc[0, 0]))
    #         if not maxdd_row.empty:
    #             maxdd_vals.append(float(maxdd_row.iloc[0, 0]))

    #     summary = {
    #         "n_samples_evaluated": len(all_metrics),
    #         "sharpe_mean": (
    #             float(pd.Series(sharpe_vals).mean()) if sharpe_vals else None
    #         ),
    #         "sharpe_std": float(pd.Series(sharpe_vals).std()) if sharpe_vals else None,
    #         "cagr_mean": float(pd.Series(cagr_vals).mean()) if cagr_vals else None,
    #         "max_dd_mean": float(pd.Series(maxdd_vals).mean()) if maxdd_vals else None,
    #         "eval_config": {
    #             "risk_free_rate": self.config.risk_free_rate,
    #             "rebalance_days": self.config.rebalance_days,
    #             "transaction_cost_rate": self.config.transaction_cost_rate,
    #             "mu_sigma_method": self.config.mu_sigma_method,
    #             "paths_sim": self.config.paths_sim,
    #             "steps_sim": self.config.steps_sim,
    #         },
    #     }

    #     with open(os.path.join(eval_dir, "eval_summary.json"), "w") as f:
    #         json.dump(summary, f, indent=4)

    #     # 4. mean weight bar chart
    #     genai_weights = [w["genai"] for w in all_weights if "genai" in w]
    #     gbm_weights = [w["gbm"] for w in all_weights if "gbm" in w]

    #     if genai_weights and gbm_weights:
    #         mean_genai = pd.DataFrame(genai_weights).mean()
    #         mean_gbm = pd.DataFrame(gbm_weights).mean()
    #         n_assets = len(mean_genai)
    #         x = range(n_assets)

    #         fig, ax = plt.subplots(figsize=(max(8, n_assets), 4))
    #         width = 0.35
    #         ax.bar(
    #             [i - width / 2 for i in x], mean_genai, width, label="GenAI", alpha=0.8
    #         )
    #         ax.bar([i + width / 2 for i in x], mean_gbm, width, label="GBM", alpha=0.8)
    #         ax.set_xticks(list(x))
    #         ax.set_xticklabels([f"A{i}" for i in x], fontsize=9)
    #         ax.set_ylabel("Mean Weight")
    #         ax.set_title("Mean Portfolio Weights — GenAI vs GBM")
    #         ax.legend()
    #         ax.grid(True, axis="y", alpha=0.3)
    #         plt.tight_layout()
    #         self.save_plot(fig, "eval_mean_weights.png")

    #     self.logger.info(
    #         f"Evaluation results saved — {len(all_metrics)} samples | "
    #         f"mean Sharpe {summary['sharpe_mean']}"
    #     )
    #     print(f"\nEval summary → {eval_dir}")
    #     print(json.dumps(summary, indent=2))
