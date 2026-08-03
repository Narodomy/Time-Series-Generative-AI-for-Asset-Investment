import os
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
class ConditioningConfig:
    name: str = "conditioning_config"
    description: str = "Configuration for conditioning (technical indicators)"

    # Conditioning
    indicator_shift: int = 20  # Concern Leakage
    indicators: List[Indicator] = field(
        default_factory=lambda: [
            # ── Trend ─────────────────────────────────────────────────────────
            # MACD: raw values, no normalisation (already a price-difference ratio)
            Indicator(
                name="MACD",
                params={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                norm_type=None,
            ),
            # ADX: bounded [0, 100] – minmax keeps it in that range
            Indicator(
                name="ADX",
                params={"timeperiod": 14},
                norm_type="minmax",
            ),
            # Parabolic SAR: expressed as distance from close (same unit as BBANDS)
            Indicator(
                name="SAR",
                params={},
                norm_type="distance_close",
            ),
            # EMA 10 / 50 / 200: distance from close (dimensionless %)
            Indicator(
                name="EMA",
                params={"timeperiod": 10},
                norm_type="distance_close",
            ),
            Indicator(
                name="EMA",
                params={"timeperiod": 50},
                norm_type="distance_close",
            ),
            Indicator(
                name="EMA",
                params={"timeperiod": 200},
                norm_type="distance_close",
            ),
            # ── Momentum ──────────────────────────────────────────────────────
            # RSI: bounded [0, 100]
            Indicator(
                name="RSI",
                params={"timeperiod": 14},
                norm_type="minmax",
            ),
            # Stochastic %K / %D: bounded [0, 100]
            Indicator(
                name="STOCH",
                params={"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                norm_type="minmax",
            ),
            # CCI: centred around 0, minmax over rolling window
            Indicator(
                name="CCI",
                params={"timeperiod": 14},
                norm_type="minmax",
            ),
            # ── Volatility ────────────────────────────────────────────────────
            # ATR: expressed as % of close (dimensionless)
            Indicator(
                name="ATR",
                params={"timeperiod": 14},
                norm_type="minmax",
            ),
            # Bollinger Bands: distance of upper/middle/lower from close
            Indicator(
                name="BBANDS",
                params={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
                norm_type="distance_close",
            ),
            # ── Volume ────────────────────────────────────────────────────────
            # OBV: unbounded cumulative – normalise with minmax
            Indicator(
                name="OBV",
                params={},
                norm_type="minmax",
            ),
            # CMF: bounded [-1, 1] – minmax is fine
            Indicator(
                name="ADOSC",  # TA-Lib's Chaikin A/D Oscillator ≈ CMF proxy
                params={"fastperiod": 3, "slowperiod": 10},
                norm_type="minmax",
            ),
            # VWAP: not a native TA-Lib function; handled separately below
            # (see add_vwap helper)
        ]
    )

    use_cyclical_time: bool = (
        True  # Whether to add cyclical features for time (e.g., day of week, month of year)
    )
    cyclical_time: str = field(
        default="weekday"
    )  # Options: 'weekday', 'month', 'day_of_month', 'hour_of_day', etc.


@dataclass
class DataEngineeringConfig:
    name: str = "data_engineering_config"
    description: str = "Configuration for data engineering"
    group_name: str = "set_index"
    subgroup_name: str = "set_idx50"

    random_seed: int = 42
    symbols: list = field(
        default_factory=lambda: [
            "ADVANC.BK",
            "AOT.BK",
            "BANPU.BK",
            "BBL.BK",
            "BCH.BK",
            "BDMS.BK",
            "BEM.BK",
            "BGRIM.BK",
            "BH.BK",
            "BJC.BK",
            "BTS.BK",
            "CBG.BK",
            "CENTEL.BK",
            "CPALL.BK",
            "CPF.BK",
            "CPN.BK",
            "DELTA.BK",
            "EA.BK",
            "EGCO.BK",
            "GLOW.BK",
            "GPSC.BK",
            "GULF.BK",
            "HMPRO.BK",
            "INTUCH.BK",
            "IRPC.BK",
            "IVL.BK",
            "KBANK.BK",
            "KCE.BK",
            "KKP.BK",
            "KTB.BK",
            "KTC.BK",
            "LH.BK",
            "MINT.BK",
            "MTC.BK",
            "PTG.BK",
            "PTT.BK",
            "PTTEP.BK",
            "PTTGC.BK",
            "RATCH.BK",
            "ROBINS.BK",
            "SCB.BK",
            "SCC.BK",
            "SPALI.BK",
            "SPRC.BK",
            "TCAP.BK",
            "TISCO.BK",
            "TTB.BK",
            "TOA.BK",
            "TOP.BK",
            "TRUE.BK",
        ]
    )
    interval: str = "1d"

    # Date range for training, validation, and testing
    train_period: tuple = ("2010-01-01", "2018-12-31")
    val_period: tuple = ("2019-01-01", "2020-12-31")
    test_period: tuple = ("2021-01-01", "2025-12-31")

    # DataLoader
    batch_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            "train": 64,
            "val": 64,
            "test": 1,
        }
    )

    # Windowing for feature engineering (e.g., for indicators that require a lookback period)
    window_size: int = 20
    window_stride: int = 1

    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    scaler_x: str = (
        "standard"  # Options: 'standard', 'robust', 'minmax', 'annual_seasonal', None
    )
    scaler_cond: str = "robust"  # Options: 'standard', 'robust', 'minmax', None

    @property
    def save_dir(self) -> Path:
        if self.group_name:
            path = PROCESSED_DIR / self.group_name / self.name
            if self.subgroup_name:
                path = PROCESSED_DIR / self.group_name / self.subgroup_name / self.name
            else:
                path = PROCESSED_DIR / self.group_name / self.name
        else:
            path = PROCESSED_DIR / self.name
        return path

    @property
    def analysis_dir(self) -> Path:
        return self.save_dir / "analysis"

    @property
    def analysis_table_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def analysis_figure_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def window_stride_non_overlapping(self):
        return self.window_size

    @property
    def indicator_shift(self):
        return (
            self.window_size
        )  # Shift indicators by the window size to prevent leakage

    @property
    def window_strides(self) -> Dict[str, int]:
        return {
            "train": self.window_stride,
            "val": self.window_stride,
            "test": self.window_stride_non_overlapping,
        }

    def __post_init__(self):
        self.name = f"{self.name}_w{self.window_size}_assets{len(self.symbols)}_scaler-x_{self.scaler_x}_scaler-cond{self.scaler_cond}"
        self.description += f" | Window Size: {self.window_size}, Indicator Shift: {self.indicator_shift}, Assets: {len(self.symbols)}"
        self.description += f" | Train: {self.train_period}, Val: {self.val_period}, Test: {self.test_period}"
        self.description += f" | Conditioning: {self.conditioning.name} with {len(self.conditioning.indicators)} indicators"

        self.conditioning.indicator_shift = self.indicator_shift
