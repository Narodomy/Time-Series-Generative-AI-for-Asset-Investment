from .helper import (
    timestamp_info,
    timestamp_mask,
    inverse_log_returns,
    load_processed_store,
)
from .statistics import (
    monte_carlo_statistic,
    simulate_mc_gbm,
    calc_expected_returns,
    calc_covariance,
    calc_volatility,
)
from .setup import setup_logging, setup_random_seed
from .analysis import (
    plot_return_dist,
    plot_compare_dist,
    plot_correlation,
    plot_rolling_vol,
    plot_compare_correlation,
    plot_compare_rolling_vol,
    _kl_divergence,
)
from .experiment import ExperimentManager
from .features import (
    inverse_x,
    inverse_cond,
    inverse_price,
    save_split,
    save_scalers,
    save_variant,
    load_split,
    load_scalers,
    load_variant,
    make_dataloader,
    make_all_dataloaders,
)

from .scaler import (
    AnnualSeasonalScaler,
    create_scaler,
    transform_windows_x,
    inverse_transform_windows_x,
    transform_windows_cond,
    inverse_transform_windows_cond,
    compute_start_indices,
)

from .datasets import WindowDataset, make_dataloaders

from .io import (
    save_data,
    load_data,
    build_datasets,
    load_and_make_dataloaders,
    verify_roundtrip,
)

__version__ = "0.2.0"
__all__ = [
    "load_processed_store",
    # Analysis
    "plot_return_dist",
    "plot_compare_dist",
    "plot_correlation",
    "plot_rolling_vol",
    "plot_compare_correlation",
    "plot_compare_rolling_vol",
    "_kl_divergence",
    "wasserstein_distance",
    # Setup
    "setup_logging",
    "setup_random_seed",
    # Helper
    "timestamp_info",
    "timestamp_mask",
    "inverse_log_returns",
    # Experiment
    "ExperimentManager",
    # Statistics
    "monte_carlo_statistic",
    "simulate_mc_gbm",
    # Features
    "inverse_x",
    "inverse_cond",
    "inverse_price",
    "save_split",
    "save_scalers",
    "save_variant",
    "load_split",
    "load_scalers",
    "load_variant",
    "make_dataloader",
    "make_all_dataloaders",
    # New Version
    "AnnualSeasonalScaler",
    "create_scaler",
    "transform_windows_x",
    "inverse_transform_windows_x",
    "transform_windows_cond",
    "inverse_transform_windows_cond",
    "WindowDataset",
    "make_dataloader",
    "save_data",
    "load_data",
    "build_datasets",
    "load_and_make_dataloaders",
    "verify_roundtrip",
    "compute_start_indices",
]
