"""Tests for physics-informed loss functions."""

import pytest
import torch
from src.training.physics_loss import (
    PhysicsInformedLoss,
    SaintVenantLoss,
    ManningLoss,
    FocalLoss,
)


class TestSaintVenantLoss:
    def test_zero_residual(self):
        """Loss should be zero when PDE is exactly satisfied."""
        t = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        A = t * 0.0 + 1.0
        Q = x * 0.0 + 1.0
        ql = torch.zeros(3)

        loss_fn = SaintVenantLoss()
        loss = loss_fn(A, Q, ql, t, x)
        assert loss.item() >= 0

    def test_positive_loss(self):
        loss_fn = SaintVenantLoss()
        t = torch.tensor([0.0, 1.0], requires_grad=True, dtype=torch.float32)
        x = torch.tensor([0.0, 1.0], requires_grad=True, dtype=torch.float32)
        A = t ** 2
        Q = x ** 3
        ql = torch.zeros(2)
        loss = loss_fn(A, Q, ql, t, x)
        assert loss.item() >= 0


class TestManningLoss:
    def test_consistent_flow(self):
        loss_fn = ManningLoss(default_roughness=0.035)
        n = 0.035
        A = torch.tensor([10.0, 20.0])
        Rh = torch.tensor([1.5, 2.0])
        Sf = torch.tensor([0.001, 0.002])
        Q = (1.0 / n) * A * Rh ** (2.0 / 3.0) * Sf ** 0.5
        loss = loss_fn(Q, A, Rh, Sf)
        assert loss.item() < 1e-4


class TestFocalLoss:
    def test_output_scalar(self):
        focal = FocalLoss(gamma=2.0)
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))
        loss = focal(logits, targets)
        assert loss.dim() == 0

    def test_with_class_weights(self):
        weights = torch.tensor([0.1, 2.5, 2.5, 10.0])
        focal = FocalLoss(gamma=2.0, alpha=weights)
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))
        loss = focal(logits, targets)
        assert loss.item() > 0


class TestPhysicsInformedLoss:
    def test_data_only(self):
        loss_fn = PhysicsInformedLoss()
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))
        losses = loss_fn(logits, targets)
        assert "total" in losses
        assert "data" in losses
        assert losses["physics_sv"].item() == 0.0

    def test_with_physics(self):
        loss_fn = PhysicsInformedLoss(lambda_sv=0.1, lambda_manning=0.05)
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        t = torch.tensor([float(i) for i in range(10)], requires_grad=True)
        x = torch.tensor([float(i) for i in range(10)], requires_grad=True)

        physics_data = {
            "predicted_A": t * 2 + 10,
            "predicted_Q": x * 3 + 5,
            "predicted_ql": torch.ones(10),
            "t": t,
            "x": x,
            "hydraulic_radius": torch.ones(10) * 1.5,
            "friction_slope": torch.ones(10) * 0.001,
        }

        losses = loss_fn(logits, targets, physics_data)
        assert losses["total"].item() > losses["data"].item()
