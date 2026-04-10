"""
Variational Quantum Circuit (VQC) implementation using PennyLane.

Implements the hardware-efficient ansatz with angle encoding
as described in Section III of the paper.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


def create_quantum_device(n_qubits: int, shots: int = None):
    """Create a PennyLane quantum device.

    Args:
        n_qubits: Number of qubits in the register.
        shots: Number of measurement shots. None for analytic (statevector).
    """
    return qml.device("default.qubit", wires=n_qubits, shots=shots)


def angle_encoding(features, wires):
    """Encode classical features as R_Y rotations (Eq. 6 in paper).

    Args:
        features: Tensor of shape (n_qubits,) with values in [-pi, pi].
        wires: Qubit wire indices.
    """
    for i, wire in enumerate(wires):
        qml.RY(features[i], wires=wire)


def hardware_efficient_ansatz(params, wires, n_layers):
    """Hardware-efficient variational ansatz (Eq. 7-8 in paper).

    Alternating R_Y, R_Z rotations with nearest-neighbor CNOT entanglement.

    Args:
        params: Tensor of shape (n_layers, n_qubits, 2) for R_Y and R_Z angles.
        wires: Qubit wire indices.
        n_layers: Number of variational layers L.
    """
    n_qubits = len(wires)

    for layer in range(n_layers):
        for i, wire in enumerate(wires):
            qml.RY(params[layer, i, 0], wires=wire)
            qml.RZ(params[layer, i, 1], wires=wire)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


class VariationalQuantumCircuit(nn.Module):
    """Variational Quantum Circuit as a PyTorch module.

    Implements the quantum processing layer of the HQC-PINN architecture.
    Uses angle encoding for feature input and hardware-efficient ansatz
    for trainable unitary, with Pauli-Z measurements on all qubits.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        shots: Measurement shots for uncertainty quantification. None = analytic.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, shots: int = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = 2 * n_qubits * n_layers
        self.shots = shots

        dev = create_quantum_device(n_qubits, shots=shots)
        self.wires = list(range(n_qubits))

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, params):
            angle_encoding(inputs, self.wires)
            reshaped = params.reshape(n_layers, n_qubits, 2)
            hardware_efficient_ansatz(reshaped, self.wires, n_layers)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

        self.params = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum circuit.

        Args:
            x: Input tensor of shape (batch_size, n_qubits) in [-pi, pi].

        Returns:
            Tensor of shape (batch_size, n_qubits) with expectation values.
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            result = self.circuit(x[i], self.params)
            outputs.append(torch.stack(result))

        return torch.stack(outputs)

    def forward_with_shots(self, x: torch.Tensor, n_shots: int = 200) -> dict:
        """Forward pass with multiple measurement shots for UQ.

        Returns mean predictions and shot-level statistics for
        uncertainty quantification (Section IV of paper).

        Args:
            x: Input tensor of shape (batch_size, n_qubits).
            n_shots: Number of measurement repetitions.

        Returns:
            Dictionary with 'mean', 'std', 'all_shots' tensors.
        """
        dev_shots = create_quantum_device(self.n_qubits, shots=1)

        @qml.qnode(dev_shots, interface="torch", diff_method="parameter-shift")
        def shot_circuit(inputs, params):
            angle_encoding(inputs, self.wires)
            reshaped = params.reshape(self.n_layers, self.n_qubits, 2)
            hardware_efficient_ansatz(reshaped, self.wires, self.n_layers)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        batch_size = x.shape[0]
        all_shots = torch.zeros(batch_size, n_shots, self.n_qubits)

        for i in range(batch_size):
            for s in range(n_shots):
                result = shot_circuit(x[i], self.params)
                all_shots[i, s] = torch.stack(result)

        mean = all_shots.mean(dim=1)
        std = all_shots.std(dim=1)

        return {"mean": mean, "std": std, "all_shots": all_shots}

    def gate_count(self) -> dict:
        """Return quantum resource estimates as described in paper."""
        n_ry = self.n_qubits * self.n_layers
        n_rz = self.n_qubits * self.n_layers
        n_cnot = (self.n_qubits - 1) * self.n_layers
        n_encoding = self.n_qubits

        return {
            "encoding_gates": n_encoding,
            "rotation_gates": n_ry + n_rz,
            "cnot_gates": n_cnot,
            "total_gates": n_encoding + n_ry + n_rz + n_cnot,
            "circuit_depth": self.n_layers * 3 + 1,
            "trainable_parameters": self.n_params,
        }
