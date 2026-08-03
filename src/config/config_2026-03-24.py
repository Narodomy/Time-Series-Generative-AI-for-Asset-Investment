import torch
import torch.nn as nn
from dataclasses import dataclass, field

CRITERION_MAP = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "HuberLoss": nn.HuberLoss,
}


@dataclass
class Config:
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
    steps_sim: int = 20
    paths_sim: int = 1000

    # Window
    window_size: int = 30
    stride: int = 1
    normalize_window: bool = True
    normalize_shift: int = window_size
    normalize_prefix_window: bool = True

    # Training
    epochs: int = 1000
    batch_size: int = 64

    max_grad_norm: float = 1.0
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
    # shift: int = field(init=False)  # will be set in __post_init__ based on steps_sim
    sequence_depth: int = (
        steps_sim  # set to steps_sim for future conditioning, or 1 for past conditioning
    )
    indicator_shift: int = (
        steps_sim  # shift for indicators; set to steps_sim for future conditioning, or 1 for past conditioning
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
    #     self.shift = self.steps_sim

    def get_criterion(self) -> nn.Module:
        if self.criterion not in CRITERION_MAP:
            raise ValueError(
                f"Unknown criterion: {self.criterion}. Choose from {list(CRITERION_MAP.keys())}"
            )
        return CRITERION_MAP[self.criterion]()
