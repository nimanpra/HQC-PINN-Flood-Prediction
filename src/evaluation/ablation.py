"""
Ablation study framework for HQC-PINN.

Systematically evaluates the contribution of each component
as reported in Table V of the paper.
"""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    use_physics_sv: bool = True
    use_physics_manning: bool = True
    use_quantum: bool = True
    use_entanglement: bool = True
    use_preprocessing: bool = True
    lambda_sv: float = 0.1
    lambda_manning: float = 0.05


ABLATION_CONFIGS = [
    AblationConfig(name="Full HQC-PINN"),
    AblationConfig(name="- Saint-Venant loss", use_physics_sv=False, lambda_sv=0.0),
    AblationConfig(name="- Manning loss", use_physics_manning=False, lambda_manning=0.0),
    AblationConfig(name="- Both physics losses", use_physics_sv=False, use_physics_manning=False, lambda_sv=0.0, lambda_manning=0.0),
    AblationConfig(name="- Quantum layer", use_quantum=False),
    AblationConfig(name="- Entanglement", use_entanglement=False),
    AblationConfig(name="- Pre-processing", use_preprocessing=False),
]


def run_ablation_study(
    base_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    n_epochs: int = 100,
) -> list:
    """Run the complete ablation study.

    Args:
        base_config: Base model configuration dictionary.
        train_loader: Training data.
        val_loader: Validation data.
        test_loader: Test data.
        device: Compute device.
        n_epochs: Max training epochs per ablation.

    Returns:
        List of result dictionaries for each ablation config.
    """
    results = []

    for config in ABLATION_CONFIGS:
        logger.info(f"Running ablation: {config.name}")

        from ..models.hqc_pinn import HQCPINN
        from ..models.classical_pinn import ClassicalPINN
        from ..training.physics_loss import PhysicsInformedLoss
        from ..training.trainer import HQCPINNTrainer

        if not config.use_quantum:
            model = ClassicalPINN(
                input_dim=base_config["input_dim"],
                n_classes=base_config["n_classes"],
            )
        else:
            model = HQCPINN(
                input_dim=base_config["input_dim"],
                n_qubits=base_config["n_qubits"],
                n_layers=base_config["n_layers"],
                n_classes=base_config["n_classes"],
            )

        loss_fn = PhysicsInformedLoss(
            lambda_sv=config.lambda_sv,
            lambda_manning=config.lambda_manning,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = HQCPINNTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            save_dir=f"results/ablation/{config.name.replace(' ', '_')}",
        )

        history = trainer.train(train_loader, val_loader, n_epochs=n_epochs, patience=10)
        val_metrics = trainer.validate(test_loader)

        result = {
            "config": config.name,
            "accuracy": val_metrics["accuracy"],
            "convergence_epoch": history.get("convergence_epoch"),
            "total_epochs": history["total_epochs"],
            "best_val_loss": history["best_val_loss"],
            "n_parameters": sum(p.numel() for p in model.parameters()),
        }

        results.append(result)
        logger.info(f"  -> Accuracy: {result['accuracy']:.4f}, Epochs: {result['total_epochs']}")

    return results
