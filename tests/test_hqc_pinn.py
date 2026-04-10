"""Tests for the HQC-PINN model."""

import pytest
import torch
from src.models.hqc_pinn import HQCPINN


class TestHQCPINN:
    """Test suite for the full HQC-PINN model."""

    def test_forward_shape(self):
        model = HQCPINN(input_dim=25, n_qubits=4, n_layers=2, n_classes=4)
        x = torch.randn(3, 25)
        out = model(x)
        assert out.shape == (3, 4)

    def test_parameter_count(self):
        model = HQCPINN(input_dim=25, n_qubits=8, n_layers=3, n_classes=4)
        info = model.count_parameters()
        assert info["quantum_vqc"] == 48
        assert info["total"] == info["classical_pre"] + info["quantum_vqc"] + info["classical_post"]

    def test_parameter_reduction_vs_classical(self):
        """Verify HQC-PINN uses fewer parameters than classical baseline."""
        from src.models.classical_pinn import ClassicalPINN

        hqc = HQCPINN(input_dim=25, n_qubits=8, n_layers=3, n_classes=4)
        cpinn = ClassicalPINN(input_dim=25, hidden_dims=[256, 128, 64], n_classes=4)

        hqc_params = hqc.count_parameters()["total"]
        cpinn_params = cpinn.count_parameters()

        assert hqc_params < cpinn_params

    def test_predict_with_uncertainty(self):
        model = HQCPINN(input_dim=25, n_qubits=4, n_layers=2, n_classes=4)
        x = torch.randn(2, 25)
        result = model.predict_with_uncertainty(x, n_shots=5)

        assert result["logits_mean"].shape == (2, 4)
        assert result["entropy"].shape == (2,)
        assert result["aleatoric_uncertainty"].shape == (2,)
        assert (result["entropy"] >= 0).all()

    def test_gradient_flow_full_model(self):
        model = HQCPINN(input_dim=25, n_qubits=4, n_layers=2, n_classes=4)
        x = torch.randn(2, 25)
        y = torch.tensor([0, 1])

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
