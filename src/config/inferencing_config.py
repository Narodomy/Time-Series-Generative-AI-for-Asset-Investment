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

_OPTUNA_KEY_MAP = {
    "n_layers": "num_layers",
    "n_heads": "num_attention_heads",
    # dim_feedforward, best_val_loss, lr, weight_decay
    # — lr/weight_decay ไม่ใช่ field ของ ModelArchConfig จะถูก filter ออกเองใน valid
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
    subgroup: str = ""
    description: str = "Configuration for inference process"

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    random_seed: int = 78

    processed_root: Path = field(
        default_factory=lambda: Path(PROCESSED_DIR)
    )  # Processed Data root (Input)
    processed_group_name: str = ""
    processed_subgroup_name: str = ""
    processed_name: str = ""
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
    def processed_dir(self) -> Path:
        """Full feature-store path: processed_root / group / subgroup / name."""
        parts = [
            p
            for p in (self.processed_group_name, self.processed_subgroup_name)
            if p != ""
        ]
        return self.processed_root.joinpath(*parts, self.processed_name)

    @property
    def experiment_dir(self) -> Path:
        """Mirrors ExperimentConfig.exp_dir: experiments_dir / group / subgroup / experiment_name."""
        if self.group:
            if self.subgroup:
                return (
                    self.experiments_dir
                    / self.group
                    / self.subgroup
                    / self.experiment_name
                )
            return self.experiments_dir / self.group / self.experiment_name
        return self.experiments_dir / self.experiment_name

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

    def load_processed_info(self) -> tuple[str, str, str]:
        """Returns (group_name, subgroup_name, name) from experiment_config.json (root-level,
        keys: processed_group_name / processed_subgroup_name / procesed_name)."""
        raw = self._load_experiment_config()
        return (
            raw.get("processed_group_name", ""),
            raw.get("processed_subgroup_name", ""),
            raw.get("procesed_name", ""),
        )

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
        """Returns batch_sizes from experiment_config.json → training.batch_sizes."""
        return self._load_experiment_config()["training"]["batch_sizes"]

    def load_best_params(self) -> ModelArchConfig:
        """Load model params from optuna/best_params.json.
        Keys follow Optuna naming (n_layers, n_heads) — remapped to ModelArchConfig fields.
        Raises FileNotFoundError if file doesn't exist."""
        path = self.experiment_dir / "optuna" / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(f"best_params.json not found: {path}")
        with open(path) as f:
            raw = json.load(f)

        # remap Optuna keys → dataclass field names
        remapped = {_OPTUNA_KEY_MAP.get(k, k): v for k, v in raw.items()}
        valid = {f.name for f in fields(self.model_params)}

        # timesteps/beta_start/beta_end ไม่อยู่ใน best_params → ดึงจาก current model_params
        return replace(
            self.model_params,  # ← keeps timesteps, beta_start, beta_end จาก default
            **{k: v for k, v in remapped.items() if k in valid},
        )

    def apply_experiment_config(
        self,
        model_params_source: ModelParamsSource = ModelParamsSource.BEST,
    ) -> "InferenceConfig":
        """
        Fill processed_* fields and model_params from experiment_config.json.
        model_params_source:
        - BEST    → optuna/best_params.json (fallback to DEFAULT if missing)
        - DEFAULT → experiment_config.json ["model"]
        """
        group_name, subgroup_name, name = self.load_processed_info()
        self.processed_group_name = group_name
        self.processed_subgroup_name = subgroup_name
        self.processed_name = name

        if model_params_source == ModelParamsSource.BEST:
            try:
                self.model_params = self.load_best_params()
                print("[InferenceConfig] model_params loaded from best_params.json")
            except FileNotFoundError:
                self.model_params = self.load_model_params()
                print(
                    "[InferenceConfig] best_params.json not found — fallback to experiment_config.json"
                )
        else:
            self.model_params = self.load_model_params()
            print("[InferenceConfig] model_params loaded from experiment_config.json")

        return self
