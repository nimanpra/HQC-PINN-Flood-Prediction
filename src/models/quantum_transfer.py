"""
Quantum Transfer Learning (QTL) protocol.

Implements the Classical-to-Quantum (CQ) transfer learning paradigm
described in Section VI of the paper, following Mari et al. (2020).

Phase 1: Classical pre-training on multi-hazard data (frozen after training).
Phase 2: Quantum fine-tuning on flood-specific data (only VQC params trained).
"""

import torch
import torch.nn as nn

from .quantum_circuit import VariationalQuantumCircuit


class QuantumTransferLearning(nn.Module):
    """Quantum Transfer Learning model.

    Uses a pre-trained classical feature extractor (frozen) with
    a trainable variational quantum circuit as the classifier.

    Args:
        pretrained_backbone: Pre-trained classical network (will be frozen).
        backbone_out_dim: Output dimension of the backbone.
        n_qubits: Number of qubits in the VQC.
        n_layers: Number of variational layers.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        pretrained_backbone: nn.Module,
        backbone_out_dim: int = 64,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_classes: int = 4,
    ):
        super().__init__()

        self.backbone = pretrained_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(backbone_out_dim, n_qubits),
            nn.Tanh(),
        )
        self.scale = torch.pi

        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        z = self.adapter(features) * self.scale
        q_out = self.vqc(z)
        return self.classifier(q_out)

    def trainable_parameters(self) -> int:
        """Count only the trainable (non-frozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiHazardBackbone(nn.Module):
    """Classical backbone for multi-hazard pre-training (Phase 1).

    Trained on 82 multi-hazard disaster events before freezing.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
