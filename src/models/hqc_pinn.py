"""
Hybrid Quantum-Classical Physics-Informed Neural Network (HQC-PINN).

The core architecture described in Section III of the paper:
  Input -> Classical Pre-Net -> VQC -> Classical Post-Net -> Output
with physics-informed loss from Saint-Venant and Manning's equations.
"""

import torch
import torch.nn as nn
import numpy as np

from .quantum_circuit import VariationalQuantumCircuit


class ClassicalPreNet(nn.Module):
    """Classical pre-processing network (Eq. 10 in paper).

    z = pi * tanh(W_2 * ReLU(W_1 * x + b_1) + b_2)

    Two-layer network reducing d-dimensional input to n_qubits
    values in [-pi, pi] via learned compression.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64, n_qubits: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),
        )
        self.scale = np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale


class ClassicalPostNet(nn.Module):
    """Classical post-processing network.

    Maps quantum expectation values to final prediction space.
    """

    def __init__(self, n_qubits: int = 8, hidden_dim: int = 32, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HQCPINN(nn.Module):
    """Hybrid Quantum-Classical Physics-Informed Neural Network.

    Complete architecture as described in Section III, Eq. 9:
        y_hat(x) = h_psi( Q_phi( g_omega(x) ) )

    Args:
        input_dim: Dimension of multi-modal input features.
        n_qubits: Number of qubits in the VQC.
        n_layers: Number of variational layers.
        n_classes: Number of output classes (flood severity levels).
        hidden_pre: Hidden dimension for pre-processing network.
        hidden_post: Hidden dimension for post-processing network.
    """

    def __init__(
        self,
        input_dim: int = 25,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_classes: int = 4,
        hidden_pre: int = 64,
        hidden_post: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.pre_net = ClassicalPreNet(input_dim, hidden_pre, n_qubits)
        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers)
        self.post_net = ClassicalPostNet(n_qubits, hidden_post, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full HQC-PINN.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, n_classes).
        """
        z = self.pre_net(x)
        q_out = self.vqc(z)
        logits = self.post_net(q_out)
        return logits

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_shots: int = 200
    ) -> dict:
        """Prediction with quantum uncertainty quantification (Section IV).

        Uses Born-rule measurement stochasticity for inherent UQ.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            n_shots: Number of measurement shots.

        Returns:
            Dictionary with 'logits_mean', 'logits_std', 'probs_mean',
            'entropy', 'aleatoric_uncertainty'.
        """
        z = self.pre_net(x)
        shot_results = self.vqc.forward_with_shots(z, n_shots)

        batch_size = x.shape[0]
        all_logits = torch.zeros(batch_size, n_shots, self.n_classes)

        for s in range(n_shots):
            all_logits[:, s, :] = self.post_net(shot_results["all_shots"][:, s, :])

        all_probs = torch.softmax(all_logits, dim=-1)

        logits_mean = all_logits.mean(dim=1)
        logits_std = all_logits.std(dim=1)
        probs_mean = all_probs.mean(dim=1)

        # Predictive entropy (Eq. 16)
        entropy = -(probs_mean * torch.log(probs_mean + 1e-10)).sum(dim=-1)

        # Aleatoric uncertainty (Eq. 14)
        aleatoric = all_probs.var(dim=1).mean(dim=-1)

        return {
            "logits_mean": logits_mean,
            "logits_std": logits_std,
            "probs_mean": probs_mean,
            "entropy": entropy,
            "aleatoric_uncertainty": aleatoric,
        }

    def count_parameters(self) -> dict:
        """Count parameters by component (Table III in paper)."""
        pre_params = sum(p.numel() for p in self.pre_net.parameters())
        quantum_params = self.vqc.n_params
        post_params = sum(p.numel() for p in self.post_net.parameters())

        return {
            "classical_pre": pre_params,
            "quantum_vqc": quantum_params,
            "classical_post": post_params,
            "total": pre_params + quantum_params + post_params,
            "quantum_gate_info": self.vqc.gate_count(),
        }
