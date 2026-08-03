import json
import torch
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, fields, replace
from utils.paths import EXPERIMENTS_DIR, PROCESSED_DIR
from .modeling_config import DDPMTransformerConfig, ModelArchConfig

_KEY_MAP = {
    "n_layers": "num_layers",
    "n_heads": "num_attention_heads",
}


class ModelParamsSource(Enum):
    DEFAULT = "default"  # exp_dir/config.json
    BEST = "best"  # exp_dir/optuna/best_params.json


@dataclass
class InjectionConfig:
    feat_names: list[str] = field(
        default_factory=lambda: [
            "ATR",  # f=0
            "BB_lower",  # f=1
            "BB_mid",  # f=2  ← PC1 top
            "BB_upper",  # f=3
            "MACD",  # f=4
            "MACD_hist",  # f=5  ← PC1 #2
            "MACD_sig",  # f=6
            "ROC",  # f=7
            "RSI",  # f=8
        ]
    )

    scenarios: list[dict] = field(
        default_factory=lambda: [
            ("f=2 BB_mid", [2]),
            ("f=5 MACD_hist", [5]),
            ("f=8 RSI", [8]),
            ("f=0 ATR", [0]),
            ("f=7 ROC", [7]),
            ("f=1,2,3 all_BBANDS", [1, 2, 3]),
            ("f=4,5,6 all_MACD", [4, 5, 6]),
            ("f=0..8 ALL", list(range(9))),
        ]
    )

    values: list[float] = field(default_factory=lambda: [-5000, 10000])
    cond_window_shift: list[int] = field(default_factory=lambda: [1, 2, 3, 5])


@dataclass
class InferenceConfig:
    name: str = "inference_config"
    group: str = "default"
    description: str = "Configuration for inference process"

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    random_seed: int = 78

    processed_dir: Path = field(
        default_factory=lambda: Path(PROCESSED_DIR)
    )  # Processed Data (Input)
    experiments_dir: Path = field(
        default_factory=lambda: Path(EXPERIMENTS_DIR)
    )  # Experimental Model (Input)
    experiment_name: str = ""
    simulation_name: str = ""
    checkpoint_file: str = ""

    # Model Params
    model_params: ModelArchConfig = field(default_factory=DDPMTransformerConfig)
    # Monte Carlo Simulation
    num_simulations: int = 1000

    @property
    def experiment_dir(self) -> Path:
        if self.group == "":
            return self.experiments_dir / self.experiment_name
        return self.experiments_dir / self.group / self.experiment_name

    @property
    def simulation_dir(self) -> Path:
        return self.experiment_dir / "simulations" / self.simulation_name

    @property
    def checkpoint_path(self) -> Path:
        return self.experiment_dir / "checkpoints" / self.checkpoint_file

    @property
    def experiment_config_path(self) -> Path:
        return self.experiment_dir / "experiment_config.json"

    def _load_experiment_config(self) -> dict:
        path = self.experiment_config_path
        if not path.exists():
            raise FileNotFoundError(f"experiment_config.json not found: {path}")
        with open(path) as f:
            return json.load(f)

    def load_feature_store_info(self) -> tuple[str, str]:
        """Returns (processed_name, variant) from experiment_config.json → feature_store."""
        raw = self._load_experiment_config()
        fs = raw["feature_store"]
        return fs["name"], fs["variant"]

    # def load_data_config(self) -> DataConfig:
    #     """Returns DataConfig from experiment_config.json → exp_cfg.data."""
    #     raw = self._load_experiment_config()["exp_cfg"]["data"]
    #     valid = {f.name for f in fields(DataConfig)}
    #     # normalize string → tuple for any tuple field
    #     kwargs = {}
    #     for f in fields(DataConfig):
    #         if f.name not in raw:
    #             continue
    #         val = raw[f.name]
    #         if isinstance(f.default, tuple) or (
    #             hasattr(f, "default_factory") and isinstance(f.default_factory(), tuple)  # type: ignore
    #         ):
    #             val = (val,) if isinstance(val, str) else tuple(val)
    #         kwargs[f.name] = val
    #     return DataConfig(**kwargs)

    def load_batch_sizes(self) -> dict[str, int]:
        """Returns batch_sizes from experiment_config.json → exp_cfg.training.batch_sizes."""
        return self._load_experiment_config()["exp_cfg"]["training"]["batch_sizes"]

    def load_model_params(self) -> ModelArchConfig:
        raw = self._load_experiment_config()["exp_cfg"]["model"]
        remapped = {_KEY_MAP.get(k, k): v for k, v in raw.items()}
        valid = {f.name for f in fields(self.model_params)}
        return replace(
            self.model_params, **{k: v for k, v in remapped.items() if k in valid}
        )
