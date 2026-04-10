"""
Classical Physics-Informed Neural Network (cPINN) baseline.

Standard MLP with tanh activations and identical physics loss,
used as the primary classical baseline (Section VII.C of paper).
"""

import torch
import torch.nn as nn


class ClassicalPINN(nn.Module):
    """Classical PINN baseline with MLP architecture.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: List of hidden layer dimensions.
        n_classes: Number of output classes.
        activation: Activation function ('tanh' or 'relu').
    """

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dims: list = None,
        n_classes: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), act_fn])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
