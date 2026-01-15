import torch
from dataclasses import dataclass, asdict, field

__version__ = "0.1.4"
__all__ = [
    "OptimizerConfig", 
    "SchedulerConfig",
    "DDPMConfig",
    "TrainConfig",
]

@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class SchedulerConfig:
    use_scheduler: bool = False     
    type: str = 'cosine'            # "Cosine, (Type of Scheduler)"
    eta_min: float = 1e-6           # This LR Min

@dataclass
class DDPMConfig:
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    # NN Model
    # DDPM Transformer
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_t: int = 64 * 2

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig) # optim.AdamW
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig) # optim.lr_scheduler.CosineAnnealingLR
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)


