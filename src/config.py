from dataclasses import dataclass, asdict, field

__version__ = "0.1.3"
__all__ = [
    "OptimizerConfig", 
    "SchedulerConfig",
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
class TrainConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig) # optim.AdamW
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig) # optim.lr_scheduler.CosineAnnealingLR
    batch_size: int = 32
    epochs: int = 100