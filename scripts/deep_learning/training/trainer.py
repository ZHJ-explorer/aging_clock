import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import logging
import os

from ..models.base.base_model import BaseDeepModel


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: BaseDeepModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[str] = None,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 20,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-4
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.is_early_stopped = False

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_samples = 0

        all_predictions = []
        all_targets = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_x)

            loss = self.criterion(outputs, batch_y)

            if self.l1_reg > 0:
                l1_loss = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.l1_reg * l1_loss

            if self.l2_reg > 0:
                l2_loss = sum(p.pow(2).sum() for p in self.model.parameters())
                loss = loss + self.l2_reg * l2_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_mae += torch.abs(outputs - batch_y).sum().item()
            n_samples += len(batch_x)

            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        mae = total_mae / n_samples
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'loss': total_loss / n_samples,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_samples = 0

        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * len(batch_x)
                total_mae += torch.abs(outputs - batch_y).sum().item()
                n_samples += len(batch_x)

                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        mae = total_mae / n_samples
        rmse = np.sqrt(total_loss / n_samples)

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'loss': total_loss / n_samples,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        return self.validate(data_loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        callbacks: Optional[List[Callable]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_trainable_params()}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train MAE: {train_metrics['mae']:.4f}, Val MAE: {val_metrics['mae']:.4f}, "
                    f"Train R2: {train_metrics['r2']:.4f}, Val R2: {val_metrics['r2']:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                if save_dir:
                    self.model.save(os.path.join(save_dir, 'best_model.pt'))
                logger.info(f"New best model saved with val_loss: {val_metrics['loss']:.4f}")
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
                    callback(epoch, train_metrics['loss'], val_metrics['loss'])

        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, 'last_checkpoint.pt'))

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
