import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from utils.paths import PROCESSED_DIR

CRITERION_MAP = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "HuberLoss": nn.HuberLoss,
}


@dataclass
class Indicator:
    name: str
    params: Dict = field(default_factory=dict)
    norm_type: Optional[str] = None


@dataclass
class FeatureConfig:
    symbols: list = field(
        default_factory=lambda: [
            "AAPL",
            "TSLA",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "GOOG",
            "META",
            "AVGO",
            "ORCL",
            "CRM",
            "ADBE",
            "AMD",
            "CSCO",
        ]
    )
    interval: str = "1d"
    start_date: str = "2010-01-01"
    end_date: str = "2023-01-01"

    test_start_date: str = "2024-01-01"
    test_end_date: str = "2025-12-31"
    target: str = "Log_Returns"

    # Sequencing
    sequence_mode: str = "Backward"
    sequence_depth: int = 20  # Set to horizon for future conditioning

    # Conditioning
    indicators: List[Indicator] = field(
        default_factory=lambda: [
            Indicator(name="RSI", params={"timeperiod": 14}, norm_type="minmax"),
            Indicator(
                name="SMA", params={"timeperiod": 20}, norm_type="distance_close"
            ),
            Indicator(
                name="MACD",
                params={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                norm_type=None,
            ),
            Indicator(
                name="BBANDS",
                params={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
                norm_type="distance_close",
            ),
        ]
    )
    indicator_period: int = 14
    indicator_shift: int = 20  # Set to horizon for future conditioning


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 64
    batch_size_test: int = 1
    max_grad_norm: float = 1.0
    clip_gradients: bool = True
    save_every_epochs: int = 50

    # Optimizer
    criterion: str = "MSELoss"
    # lr: float = 1e-4
    lr: float = 1e-5
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    eta_min: float = 1e-6

    # Step LR
    step_size: int = 100
    step_gamma: float = 0.5

    # Cosine Restart
    T_0: int = 100
    T_mult: int = 1

    # Warmup
    warmup_epochs: int = 10

    def get_criterion(self) -> nn.Module:
        if self.criterion not in CRITERION_MAP:
            raise ValueError(
                f"Unknown criterion: {self.criterion}. Choose from {list(CRITERION_MAP.keys())}"
            )
        return CRITERION_MAP[self.criterion]()

    def get_scheduler(self, optimizer, total_epochs: int, scheduler_type: str = None):
        stype = scheduler_type if scheduler_type is not None else self.scheduler_type
        if not self.use_scheduler and scheduler_type is None:
            return None

        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=self.eta_min
            )
        elif stype == "cosine_restart":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min
            )
        elif stype == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.step_gamma
            )
        elif stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, min_lr=self.eta_min
            )
        elif stype == "warmup_cosine":
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - self.warmup_epochs, eta_min=self.eta_min
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")


@dataclass
class ModelConfig:
    model_name: str = (
        "DDPM_Transformer"  # "DDPM_Transformer" or "Flow Matching" or "DDIM_Transformer"
    )

    # DDPM
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # DDPM Transformer
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_t: int = 128 * 2
    dropout: float = 0.1
    dim_feedforward: int = 512


@dataclass
class InferenceConfig:
    horizon: int = 30
    n_paths: int = 1000
    window_size: int = 60  # 30, 60
    stride: int = 1
    test_batch_stride: int = 60  # Default to window_size
    normalize_window: bool = True
    normalize_shift: int = 30  # Default to window_size


@dataclass
class OptunaConfig:
    n_trials: int = 100
    min_resource: int = 3
    max_resource: int = 50
    reduction_factor: int = 3
    epochs_per_trial: int = 50

    # dim_feedforward must be >= d_model (2 times d_model is common);
    # n_heads must divide d_model;
    # noise_steps should be in [100, 2000] for reasonable training times

    suggest_lr: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3])
    suggest_d_model: List[int] = field(default_factory=lambda: [256, 512, 1024])
    suggest_ff_mult: List[int] = field(
        default_factory=lambda: [2, 4]
    )  # Multiplier for dim_feedforward relative to d_model
    suggest_n_heads: List[int] = field(default_factory=lambda: [2, 4, 8])
    suggest_n_layers: List[int] = field(default_factory=lambda: [2, 4, 6])
    # suggest_dim_feedforward: List[int] = field(
    #     default_factory=lambda: [1024, 2048, 4096]
    # )
    suggest_dropout: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    suggest_noise_steps: List[int] = field(
        default_factory=lambda: [1000, 1250, 1500, 1750]
    )
    suggest_beta_start: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    suggest_beta_end: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    suggest_weight_decay: List[float] = field(
        default_factory=lambda: [1e-7, 1e-6, 1e-5, 1e-4]
    )
    suggest_scheduler_type: List[str] = field(
        default_factory=lambda: ["cosine", "warmup_cosine", "plateau"]
    )


@dataclass
class Config:
    name: str = "v2_baseline"
    description: str = (
        "DDPM Transformer inside laten space from [B, A, T, C] when compute [B, T, A * C]"
    )

    processed_name: str = "trial21_w30h30a14rand78"
    save_processed_dir: Path = PROCESSED_DIR

    skip_training: bool = False
    skip_optuna: bool = False
    skip_evaluation: bool = False
    skip_setup_random_seed: bool = False

    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_ratios: list = field(default_factory=lambda: [0.8, 0.1, 0.1])
    random_seed: int = 78

    feature: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):

        # self.inference.normalize_shift = self.inference.window_size
        self.inference.test_batch_stride = self.inference.window_size


@dataclass
class PortfolioConfig:
    annual_risk_free_rate: float = 0.02  # rf (annualised 2 %)
    rebalance_frequency: int = 5  # rebalance every N trading days
    optimize_window: int = 60  # use past N days of returns for optimization
    transaction_cost_rate: float = 0.0025  # 25 bps one-way turnover cost
    mu_sigma_method: str = "Per-Path"  # Literal["Per-Path", "Pooled"]
    init_cash: float = 1.0
    is_log_return: bool = True
    save_reports: bool = True  # save QuantStats HTML reports
    benchmark_strategy: str = "gbm"  # "gbm" or "equal_weight"
    eval_methods: list = field(
        default_factory=lambda: [
            "Per-Path",
            "Pooled",
            "Combined",
            "Separate_Mean",
            "Separate_Median",
            "Separate_Top_K",
        ]
    )
    top_k: int = 50


@dataclass
class DefaultConfig:
    exp_name: str = "v1_baseline_test_product"
    description: str = (
        "DDPM Transformer inside laten space from [B, L, A, C] when compute [B, L, A * C]"
    )
    skip_training: bool = False
    skip_optuna: bool = False
    skip_evaluation: bool = False  # set True to skip evaluation phase
    skip_setup_random_seed: bool = (
        False  # set True to skip random seed setup (for reproducibility)
    )

    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 78
    dataset_ratios: list = field(default_factory=lambda: [0.8, 0.1, 0.1])

    # Simulation
    horizon: int = 20
    n_paths: int = 1000

    # Window
    window_size: int = 30
    stride: int = 1
    normalize_window: bool = True
    normalize_shift: int = field(default_factory=lambda: 30)
    normalize_prefix_window: bool = True

    # Training
    epochs: int = 1000
    batch_size: int = 64

    # max_grad_norm: float = 1.0
    max_grad_norm: float = 10.0
    clip_gradients: bool = True
    save_every_epochs: int = 50

    # Optimizer
    criterion: str = "MSELoss"
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    eta_min: float = 1e-6

    # Model
    model_name: str = (
        "DDPM_Transformer"  # "DDPM_Transformer" or "Flow Matching" or "DDIM_Transformer"
    )

    # DDPM
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # DDPM Transformer
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_t: int = 128 * 2
    dropout: float = 0.1
    dim_feedforward: int = 512

    # Optuna
    n_trials: int = 100
    min_resource: int = 3
    max_resource: int = 50
    reduction_factor: int = 3
    epochs_per_trial: int = 50

    # Features
    symbols: list = field(
        default_factory=lambda: [
            "AAPL",
            "TSLA",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "GOOG",
            "META",
            "AVGO",
            "ORCL",
            "CRM",
            "ADBE",
            "AMD",
            "CSCO",
        ]
    )
    interval: str = "1d"
    start_date: str = "2010-01-01"
    end_date: str = "2023-01-01"
    sequence_mode: str = "Backward"  # "Backward" or "Forward"

    # Conditioning
    indicators: list = field(default_factory=lambda: ["RSI", "SMA", "EMA"])
    indicator_period: int = 14
    # shift: int = 1 # that's the default for past conditioning, but for future conditioning we need to shift by the number of simulation steps
    # shift: int = field(init=False)  # will be set in __post_init__ based on horizon
    sequence_depth: int = (
        horizon  # set to horizon for future conditioning, or 1 for past conditioning
    )
    indicator_shift: int = (
        horizon  # shift for indicators; set to horizon for future conditioning, or 1 for past conditioning
    )

    # ------------------------------------------------------------------
    # Evaluation / Portfolio Optimisation
    # ------------------------------------------------------------------
    eval_scores: list = field(
        default_factory=lambda: [
            "r2_score",
        ]
    )

    # Risk-free rate passed to Portfolio optimizer and QuantStats.
    # Store as annual rate; divide by 252 inside Portfolio when needed.
    annual_risk_free_rate: float = 0.02  # rf (annualised 2 %)
    rebalance_frequency: int = 5  # rebalance every N trading days
    optimize_window: int = 60  # use past N days of returns for optimization
    transaction_cost_rate: float = 0.0025  # 25 bps one-way turnover cost
    benchmark_tc_rate: float = (
        0.0025  # 25 bps one-way turnover cost for benchmark (e.g., equal-weighted portfolio)
    )
    mu_sigma_method: str = "combined"  # 'combined' | 'separate'
    save_reports: bool = True  # save QuantStats HTML reports
    eval_methods: list = field(
        default_factory=lambda: [
            "combined",
            "separate_mean",
            "separate_median",
            "separate_top_k",
        ]
    )
    top_k: int = 50

    # def __post_init__(self):
    #     self.shift = self.horizon

    def get_criterion(self) -> nn.Module:
        if self.criterion not in CRITERION_MAP:
            raise ValueError(
                f"Unknown criterion: {self.criterion}. Choose from {list(CRITERION_MAP.keys())}"
            )
        return CRITERION_MAP[self.criterion]()
