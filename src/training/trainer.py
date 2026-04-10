"""
Training loop for HQC-PINN models.

Handles training, validation, early stopping, and logging
for all model variants (HQC-PINN, cPINN, VQC-only, QTL).
"""

import time
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .physics_loss import PhysicsInformedLoss

logger = logging.getLogger(__name__)


class HQCPINNTrainer:
    """Trainer for HQC-PINN and baseline models.

    Args:
        model: The model to train (HQCPINN, ClassicalPINN, etc.).
        loss_fn: Loss function (PhysicsInformedLoss or standard).
        optimizer: PyTorch optimizer.
        device: Device to train on.
        save_dir: Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: PhysicsInformedLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        save_dir: str = "results/logs",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            physics_data = batch.get("physics_data")

            self.optimizer.zero_grad()
            logits = self.model(features)
            losses = self.loss_fn(logits, targets, physics_data)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_data_loss += losses["data"].item()
            total_physics_loss += (losses["physics_sv"].item() + losses["physics_manning"].item())
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "data_loss": total_data_loss / n_batches,
            "physics_loss": total_physics_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)

            logits = self.model(features)
            losses = self.loss_fn(logits, targets)

            total_loss += losses["total"].item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        patience: int = 10,
        target_loss: float = None,
    ) -> dict:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            n_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            target_loss: Stop training when val loss reaches this target.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        convergence_epoch = None

        logger.info(f"Starting training for up to {n_epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(n_epochs):
            start_time = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["epoch_time"].append(epoch_time)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
            else:
                patience_counter += 1

            if target_loss is not None and val_metrics["loss"] <= target_loss and convergence_epoch is None:
                convergence_epoch = epoch + 1
                logger.info(f"Target loss {target_loss} reached at epoch {convergence_epoch}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.history["convergence_epoch"] = convergence_epoch
        self.history["best_val_loss"] = best_val_loss
        self.history["total_epochs"] = epoch + 1

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
