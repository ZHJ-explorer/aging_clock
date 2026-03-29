import torch
from typing import Optional, Dict, Any
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, ExponentialLR


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
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
