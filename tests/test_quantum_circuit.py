"""Tests for the Variational Quantum Circuit module."""

import pytest
import torch
import numpy as np
from src.models.quantum_circuit import VariationalQuantumCircuit


class TestVariationalQuantumCircuit:
    """Test suite for VQC implementation."""

    def test_initialization(self):
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        assert vqc.n_qubits == 4
        assert vqc.n_layers == 2
        assert vqc.n_params == 2 * 4 * 2  # 16

    def test_forward_shape(self):
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        x = torch.randn(3, 4)  # batch=3, qubits=4
        out = vqc(x)
        assert out.shape == (3, 4)

    def test_output_bounded(self):
        """Pauli-Z expectation values must be in [-1, 1]."""
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        x = torch.randn(5, 4)
        out = vqc(x)
        assert (out >= -1.0 - 1e-6).all()
        assert (out <= 1.0 + 1e-6).all()

    def test_gate_count(self):
        vqc = VariationalQuantumCircuit(n_qubits=8, n_layers=3)
        info = vqc.gate_count()
        assert info["trainable_parameters"] == 48  # 2 * 8 * 3
        assert info["encoding_gates"] == 8
        assert info["cnot_gates"] == 7 * 3  # (n_qubits - 1) * n_layers

    def test_gradient_flow(self):
        """Verify gradients flow through the quantum circuit."""
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        x = torch.randn(2, 4, requires_grad=True)
        out = vqc(x)
        loss = out.sum()
        loss.backward()
        assert vqc.params.grad is not None
        assert not torch.all(vqc.params.grad == 0)


class TestVariationalQuantumCircuitUQ:
    """Test uncertainty quantification via shot sampling."""

    def test_forward_with_shots_returns_dict(self):
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        x = torch.randn(2, 4)
        result = vqc.forward_with_shots(x, n_shots=10)
        assert "mean" in result
        assert "std" in result
        assert "all_shots" in result

    def test_shot_statistics_shape(self):
        vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        x = torch.randn(2, 4)
        result = vqc.forward_with_shots(x, n_shots=10)
        assert result["mean"].shape == (2, 4)
        assert result["std"].shape == (2, 4)
        assert result["all_shots"].shape == (2, 10, 4)
