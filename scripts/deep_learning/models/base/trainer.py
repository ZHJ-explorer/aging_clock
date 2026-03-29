import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, List, Callable, Any
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[str] = None,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 20
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.is_early_stopped = False

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> float:
        pass

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> float:
        pass

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_trainable_params()}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch+1} epochs. "
                        f"No improvement for {self.early_stopping_patience} epochs."
                    )
                    self.is_early_stopped = True
                    break

            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_loss, val_loss)

        logger.info("Training completed")
        return self.history

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'patience_counter': self.patience_counter
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.patience_counter = checkpoint['patience_counter']
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")


class DatasetWrapper(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
