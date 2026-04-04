import torch
import numpy as np
from typing import Optional, Dict, Any
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, ExponentialLR


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def build_optimizer(model, optimizer_name: str = 'adamw', learning_rate: float = 0.001,
                    weight_decay: float = 1e-4, **kwargs) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adamw':
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        return SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str = 'cosine',
                    epochs: int = 100, **kwargs):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_name == 'none':
        return None
    elif scheduler_name == 'warmup_cosine':
        warmup_epochs = kwargs.get('warmup_epochs', 10)
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            base_lr=kwargs.get('base_lr', 0.001),
            min_lr=kwargs.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
