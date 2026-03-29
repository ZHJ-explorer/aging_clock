from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DNNConfig:
    input_dim: int = 350
    hidden_dims: List[int] = None
    dropout: float = 0.3
    use_batchnorm: bool = True
    output_dim: int = 1

    activation: str = 'relu'
    output_activation: Optional[str] = None

    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    batch_size: int = 32
    epochs: int = 200
    early_stopping_patience: int = 20

    scheduler_type: str = 'cosine'
    warmup_epochs: int = 10

    l1_reg: float = 0.0
    l2_reg: float = 1e-4

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]

        self._validate()

    def _validate(self):
        assert self.input_dim > 0, "input_dim must be positive"
        assert all(d > 0 for d in self.hidden_dims), "hidden_dims must be all positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'use_batchnorm': self.use_batchnorm,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'scheduler_type': self.scheduler_type,
            'warmup_epochs': self.warmup_epochs,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
