"""
Configuration management for HQC-PINN experiments.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Model architecture
    input_dim: int = 25
    n_qubits: int = 8
    n_layers: int = 3
    n_classes: int = 4
    hidden_pre: int = 64
    hidden_post: int = 32

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-5

    # Physics loss
    lambda_sv: float = 0.1
    lambda_manning: float = 0.05
    focal_gamma: float = 2.0

    # Quantum circuit
    shots: int = 200
    diff_method: str = "parameter-shift"

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # Paths
    data_dir: str = "data/processed"
    results_dir: str = "results"
    log_dir: str = "results/logs"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**json.load(f))


def load_config(path: str = "configs/default.json") -> ExperimentConfig:
    """Load experiment configuration.

    Args:
        path: Path to JSON config file. Uses defaults if file not found.
    """
    if Path(path).exists():
        return ExperimentConfig.load(path)
    return ExperimentConfig()
