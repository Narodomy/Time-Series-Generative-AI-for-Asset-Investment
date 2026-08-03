from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from utils.paths import PROCESSED_DIR


@dataclass
class Indicator:
    name: str
    params: Dict = field(default_factory=dict)
    norm_type: Optional[str] = None


@dataclass
class FeatureConfig:
    name: str = "feature_config"
    description: str = "Configuration for feature engineering"
    save_dir: Path = PROCESSED_DIR

    random_seed: int = 42
    symbols: list = field(
        default_factory=lambda: [
            "AAPL",
            "AMZN",
            "CAT",
            "EEM",
            "GOOGL",
            "JNJ",
            "JPM",
            "KO",
            "META",
            "MSFT",
            "NVDA",
            "SPY",
            "TLT",
            "TSLA",
            "GLD",
            "XOM",
        ]
    )
    interval: str = "1d"

    # Conditioning
    indicator_shift: int = 20  # Concern Leakage
    indicators: List[Indicator] = field(
        default_factory=lambda: [
            Indicator(name="RSI", params={"timeperiod": 14}, norm_type="minmax"),
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
            Indicator(name="ATR", params={"timeperiod": 14}, norm_type="minmax"),
            Indicator(name="ROC", params={"timeperiod": 10}, norm_type="minmax"),
        ]
    )

    use_exogenous: bool = (
        True  # Whether to include exogenous features (e.g., VIX, interest rates)
    )
    use_cyclical: bool = (
        True  # Whether to add cyclical features for time (e.g., day of week, month of year)
    )
    cyclical_features: str = field(
        default_factory="weekday"
    )  # Options: 'weekday', 'month', 'day_of_month', 'hour_of_day', etc.

    # Date range for training, validation, and testing
    train_period: tuple = ("2010-01-01", "2020-12-31")
    val_period: tuple = ("2021-01-01", "2021-12-31")
    test_period: tuple = ("2022-01-01", "2025-12-31")

    # DataLoader
    batch_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            "train": 64,
            "val": 64,
            "test": 1,
        }
    )

    # Sequencing
    use_sequence: bool = True
    sequence_mode: str = "Backward"
    sequence_depth: int = 20

    # Windowing for feature engineering (e.g., for indicators that require a lookback period)
    window_size: int = 20
    window_stride: int = 1
    window_stride_non_overlap: int = 20
    window_strides: Dict[str, int] = field(default_factory=dict)

    # Scaler
    x_scaler: str = "standard"  # Options: 'standard', 'minmax', 'robust'
    x_scaler_axis: str = (
        "per_channel_global"  # Options: 'per_channel_global', 'per_channel_per_symbol', 'global'
    )
    cond_scaler: str = "robust"  # Options: 'standard', 'minmax', 'robust'
    cond_scaler_axis: str = (
        "per_feature_global"  # Options: 'per_feature_global', 'per_feature_per_symbol', 'global'
    )
    scaler_exclude_features: List[str] = field(
        default_factory=lambda: [
            "weekday_sin",
            "weekday_cos",
        ]
    )
    scaler_clip_ranges: Dict[str, tuple] = field(
        default_factory=lambda: {
            "BBANDS_lowerband_distance_close": (-0.5, 0.5),
            "BBANDS_middleband_distance_close": (-0.3, 0.3),
            "BBANDS_upperband_distance_close": (-0.5, 0.3),
        }
    )

    def __post_init__(self):
        self.window_stride_non_overlap = (
            self.window_size
        )  # For testing, we want non-overlapping windows to evaluate on distinct sequences
        self.indicator_shift = (
            self.window_stride_non_overlap
        )  # Set indicator shift to window size to avoid leakage, ensuring that indicators are computed on past data only
        if not self.window_strides:
            self.window_strides = {
                "train": self.window_stride,
                "val": self.window_stride,
                "test": self.window_stride_non_overlap,
            }
