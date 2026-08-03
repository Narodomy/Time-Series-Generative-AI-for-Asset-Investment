import json
import torch
import torch.nn as nn
import dataclasses
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass, field

from utils.paths import PROCESSED_DIR, EXPERIMENTS_DIR
from types import SimpleNamespace


# ──────────────────────────────────────────────
#  Criterion
# ──────────────────────────────────────────────
@dataclass
class CriterionConfig:
    name: str = "MSELoss"  # "MSELoss" | "L1Loss" | "HuberLoss"

    _registry = {
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "HuberLoss": nn.HuberLoss,
    }

    def build(self) -> nn.Module:
        if self.name not in self._registry:
            raise ValueError(
                f"Unknown criterion: {self.name!r}. Choose from {list(self._registry)}"
            )
        return self._registry[self.name]()


# ──────────────────────────────────────────────
#  Scheduler  (plug-and-play)
# ──────────────────────────────────────────────
@dataclass
class CosineSchedulerConfig:
    name: str = "cosine"
    eta_min: float = 1e-6

    def build(self, optimizer, total_epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=self.eta_min
        )


@dataclass
class CosineRestartSchedulerConfig:
    name: str = "cosine_restart"
    T_0: int = 100
    T_mult: int = 1
    eta_min: float = 1e-6

    def build(self, optimizer, total_epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min
        )


@dataclass
class StepSchedulerConfig:
    name: str = "step"
    step_size: int = 100
    gamma: float = 0.5

    def build(self, optimizer, total_epochs: int):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )


@dataclass
class PlateauSchedulerConfig:
    name: str = "plateau"
    patience: int = 10
    factor: float = 0.5
    eta_min: float = 1e-6

    def build(self, optimizer, total_epochs: int):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=self.factor, min_lr=self.eta_min
        )


@dataclass
class WarmupCosineSchedulerConfig:
    name: str = "warmup_cosine"
    warmup_epochs: int = 10
    eta_min: float = 1e-6

    def build(self, optimizer, total_epochs: int):
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - self.warmup_epochs, eta_min=self.eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )


SchedulerConfig = Union[
    CosineSchedulerConfig,
    CosineRestartSchedulerConfig,
    StepSchedulerConfig,
    PlateauSchedulerConfig,
    WarmupCosineSchedulerConfig,
]


# ──────────────────────────────────────────────
#  Optimizer
# ──────────────────────────────────────────────
@dataclass
class AdamConfig:
    name: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8

    def build(self, model: nn.Module):
        return torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.epsilon,
        )


@dataclass
class AdamWConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8

    def build(self, model: nn.Module):
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.epsilon,
        )


OptimizerConfig = Union[AdamConfig, AdamWConfig]


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────
@dataclass
class TrainingConfig:
    num_epochs: int = 100
    max_grad_norm: float = 1.0
    use_clip_grad: bool = True
    save_checkpoint_freq: int = 10

    batch_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            "train": 64,
            "val": 64,
            "test": 1,
        }
    )

    optimizer: OptimizerConfig = field(default_factory=AdamWConfig)
    scheduler: Optional[SchedulerConfig] = field(
        default_factory=WarmupCosineSchedulerConfig
    )
    criterion: CriterionConfig = field(default_factory=CriterionConfig)


# ──────────────────────────────────────────────
#  Model  (DDPM / DDIM / Flow Matching)
# ──────────────────────────────────────────────
@dataclass
class DDPMTransformerConfig:
    name: str = "ddpm_transformer"
    description: str = (
        "DDPM Transformer — denoising diffusion with transformer backbone"
    )

    num_layers: int = 6
    num_attention_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    ff_mult: int = 4
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    def __post_init__(self):
        self.dim_feedforward = self.d_model * self.ff_mult


@dataclass
class FMConfig:
    name: str = "flow_matching"
    description: str = "Flow Matching — ODE-based generative model (Lipman et al. 2022)"

    # ── Architecture ───────────────────────────────────────────────────────
    num_layers: int = 6
    num_attention_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    ff_mult: int = 4  # dim_feedforward = d_model * ff_mult

    # ── Flow Matching Specific ─────────────────────────────────────────────
    sigma_min: float = 1e-4
    # Minimum noise std on the conditional path.
    # ค่าเล็ก → path ใกล้ OT-straight มากขึ้น, ค่าใหญ่ → regularize มากขึ้น
    # ช่วง reasonable: [1e-5, 1e-2]

    # ── ODE Solver (Inference) ─────────────────────────────────────────────
    num_steps: int = 100
    # จำนวน integration steps ตอน sampling
    # FM ต้องการน้อยกว่า DDPM มาก: 20-100 ก็ได้ผลดี
    # ยิ่ง num_steps น้อย → เร็วขึ้น แต่ error สะสมมากขึ้น
    # ช่วง reasonable: [20, 200]

    solver: str = "euler"
    # ODE solver ที่ใช้: "euler" | "midpoint"
    # "euler"    → 1 model call/step, เร็ว, เหมาะ num_steps >= 100
    # "midpoint" → 2 model calls/step, accurate กว่า, เหมาะ num_steps < 50

    def __post_init__(self):
        self.dim_feedforward = self.d_model * self.ff_mult

        valid_solvers = ("euler", "midpoint")
        if self.solver not in valid_solvers:
            raise ValueError(
                f"Unknown solver: {self.solver!r}. Choose from {valid_solvers}"
            )


ModelArchConfig = Union[DDPMTransformerConfig, FMConfig]


# ──────────────────────────────────────────────
#  Optuna
# ──────────────────────────────────────────────
@dataclass
class OptunaConfig:
    n_trials: int = 50
    epochs_per_trial: int = 20
    min_resource: int = 5
    max_resource: int = 20
    reduction_factor: int = 3

    # ── Architecture Search Space (shared: DDPM + FM) ──────────────────────
    suggest_d_model: List[int] = field(default_factory=lambda: [128, 256, 512])
    suggest_ff_mult: List[int] = field(default_factory=lambda: [2, 4])
    suggest_n_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    suggest_n_layers: List[int] = field(default_factory=lambda: [2, 8])  # [min, max]
    suggest_dropout: List[float] = field(
        default_factory=lambda: [0.0, 0.3]
    )  # [min, max]
    suggest_lr: List[float] = field(
        default_factory=lambda: [1e-5, 1e-3]
    )  # [min, max] log
    suggest_weight_decay: List[float] = field(
        default_factory=lambda: [1e-4, 1e-1]
    )  # log

    # ── Flow Matching Specific Search Space ────────────────────────────────
    suggest_sigma_min: List[float] = field(default_factory=lambda: [1e-5, 1e-2])
    # sigma_min — log scale เพราะ range ข้ามหลาย order of magnitude
    # ค่าเล็ก (1e-5): path ตรงมาก → integrate ง่าย แต่ sensitive กว่า
    # ค่าใหญ่ (1e-2): regularize มากขึ้น → robust กว่า แต่ path โค้งขึ้น
    # ใช้: trial.suggest_float("sigma_min", *cfg.suggest_sigma_min, log=True)

    suggest_num_steps: List[int] = field(default_factory=lambda: [20, 200])
    # num_steps — ค้นหา inference steps ที่ balance quality vs speed
    # ใช้: trial.suggest_int("num_steps", *cfg.suggest_num_steps, step=10)

    suggest_solver: List[str] = field(default_factory=lambda: ["euler", "midpoint"])
    # solver — categorical choice
    # ใช้: trial.suggest_categorical("solver", cfg.suggest_solver)
    # หมายเหตุ: "midpoint" + num_steps น้อย ≈ "euler" + num_steps มาก
    #           จึงควร tune ทั้งคู่พร้อมกัน

    # ── DDPM Specific Search Space (ไว้ใช้ถ้า model เป็น DDPM) ──────────
    suggest_timesteps: List[int] = field(default_factory=lambda: [500, 1000, 2000])
    # timesteps — categorical สำหรับ DDPM
    # ใช้: trial.suggest_categorical("timesteps", cfg.suggest_timesteps)

    suggest_beta_end: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.04])
    # beta_end — categorical สำหรับ DDPM noise schedule
    # ใช้: trial.suggest_categorical("beta_end", cfg.suggest_beta_end)


# ──────────────────────────────────────────────
#  Top-level experiment config
# ──────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name: str = "experiment"
    description: str = ""
    group_name: str = "set_index"
    subgroup_name: str = "set_idx50"

    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    procesed_name: str = ""
    processed_group_name: str = ""
    processed_subgroup_name: str = ""

    skip_optuna: bool = False
    skip_training: bool = False

    model: ModelArchConfig = field(default_factory=DDPMTransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)

    def __post_init__(self):
        self.description += (
            f"Skip: [Optuna: {self.skip_optuna}, Training: {self.skip_training}]"
        )

    @property
    def processed_dir(self):
        if self.processed_group_name:
            if self.processed_subgroup_name:
                path = (
                    PROCESSED_DIR
                    / self.processed_group_name
                    / self.processed_subgroup_name
                    / self.procesed_name
                )
            else:
                path = PROCESSED_DIR / self.processed_group_name / self.procesed_name
        else:
            path = PROCESSED_DIR / self.procesed_name

        return path

    @property
    def processed(self):
        path = Path(self.processed_dir) / "meta.json"

        if not path.exists():
            raise FileNotFoundError(f"meta.json not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            processed_cfg = json.load(f)

        print(f"Loaded processed_cfg from {path} — keys: {list(processed_cfg.keys())}")

        def _to_obj(obj):
            if isinstance(obj, dict):
                return {k: _to_obj(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_obj(v) for v in obj]
            return obj

        return _to_obj(processed_cfg)

    @property
    def exp_name(self) -> str:
        name_tag = self.name.replace(" ", "_")
        model_tag = self.model.name.replace("_transformer", "").replace("_matching", "")
        scheduler_tag = self.training.scheduler.name.replace("_", "-")

        if self.skip_optuna:
            optuna_tag = ""
        else:
            optuna_tag = "optuna"

        return f"{name_tag}_{model_tag}_{scheduler_tag}_{optuna_tag}"

    @property
    def exp_dir(self) -> Path:
        if self.group_name:
            if self.subgroup_name:
                path = (
                    EXPERIMENTS_DIR
                    / self.group_name
                    / self.subgroup_name
                    / self.exp_name
                )
            else:
                path = EXPERIMENTS_DIR / self.group_name / self.exp_name
        else:
            path = EXPERIMENTS_DIR / self.exp_name

        return path

    @property
    def checkpoint_dir(self) -> Path:
        return self.exp_dir / "checkpoints"

    @property
    def figure_dir(self) -> Path:
        return self.exp_dir / "figures"

    @property
    def optuna_dir(self) -> Path:
        return self.exp_dir / "optuna"
