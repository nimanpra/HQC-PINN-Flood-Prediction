"""
Physics-informed loss functions for hydrological PDE constraints.

Implements the Saint-Venant shallow water equations and Manning's equation
as differentiable loss terms (Section III.D, Eqs. 12-14 of paper).
"""

import torch
import torch.nn as nn


class SaintVenantLoss(nn.Module):
    """Saint-Venant continuity equation residual loss (Eq. 13).

    Computes the physics loss based on the 1D shallow water continuity:
        dA/dt + dQ/dx = q_l

    The residual is evaluated at collocation points using automatic
    differentiation through the quantum-classical hybrid model.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_A: torch.Tensor,
        predicted_Q: torch.Tensor,
        predicted_ql: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Saint-Venant continuity residual.

        Args:
            predicted_A: Predicted cross-sectional area, shape (N_c,).
            predicted_Q: Predicted discharge, shape (N_c,).
            predicted_ql: Predicted lateral inflow, shape (N_c,).
            t: Time coordinates of collocation points, shape (N_c,).
            x: Spatial coordinates of collocation points, shape (N_c,).

        Returns:
            Scalar loss value (mean squared residual).
        """
        predicted_A.requires_grad_(True)
        predicted_Q.requires_grad_(True)

        dA_dt = torch.autograd.grad(
            predicted_A, t,
            grad_outputs=torch.ones_like(predicted_A),
            create_graph=True, retain_graph=True
        )[0]

        dQ_dx = torch.autograd.grad(
            predicted_Q, x,
            grad_outputs=torch.ones_like(predicted_Q),
            create_graph=True, retain_graph=True
        )[0]

        residual = dA_dt + dQ_dx - predicted_ql
        return torch.mean(residual ** 2)


class ManningLoss(nn.Module):
    """Manning's equation consistency loss (Eq. 14).

    Enforces that predicted discharge satisfies Manning's flow equation:
        Q = (1/n) * A * R_h^(2/3) * S_f^(1/2)

    Args:
        default_roughness: Default Manning's roughness coefficient n.
    """

    def __init__(self, default_roughness: float = 0.035):
        super().__init__()
        self.n = default_roughness

    def forward(
        self,
        predicted_Q: torch.Tensor,
        predicted_A: torch.Tensor,
        hydraulic_radius: torch.Tensor,
        friction_slope: torch.Tensor,
        roughness: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute Manning's equation consistency loss.

        Args:
            predicted_Q: Predicted discharge, shape (N_c,).
            predicted_A: Predicted cross-sectional area, shape (N_c,).
            hydraulic_radius: R_h values at collocation points, shape (N_c,).
            friction_slope: S_f values at collocation points, shape (N_c,).
            roughness: Manning's n at each point. Uses default if None.

        Returns:
            Scalar loss value.
        """
        if roughness is None:
            roughness = torch.full_like(predicted_Q, self.n)

        Q_manning = (
            (1.0 / roughness)
            * predicted_A
            * torch.pow(hydraulic_radius.clamp(min=1e-6), 2.0 / 3.0)
            * torch.pow(friction_slope.clamp(min=1e-6), 0.5)
        )

        residual = predicted_Q - Q_manning
        return torch.mean(residual ** 2)


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification (Eq. 12).

    Addresses severe class imbalance in flood severity data
    (No Flood: 91%, Low: 4%, Moderate: 4%, Severe: 1%).

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples.
        alpha: Per-class weights. If None, uniform weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        return focal_loss.mean()


class PhysicsInformedLoss(nn.Module):
    """Combined HQC-PINN loss function (Eq. 11).

    L_HQC = L_data + lambda_SV * L_SV + lambda_M * L_Manning

    Args:
        lambda_sv: Weight for Saint-Venant physics loss.
        lambda_manning: Weight for Manning's equation loss.
        focal_gamma: Focal loss gamma parameter.
        class_weights: Per-class weights for focal loss.
    """

    def __init__(
        self,
        lambda_sv: float = 0.1,
        lambda_manning: float = 0.05,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.lambda_sv = lambda_sv
        self.lambda_manning = lambda_manning
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.sv_loss = SaintVenantLoss()
        self.manning_loss = ManningLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        physics_data: dict = None,
    ) -> dict:
        """Compute total HQC-PINN loss.

        Args:
            logits: Model output logits, shape (batch_size, n_classes).
            targets: Ground truth labels, shape (batch_size,).
            physics_data: Optional dict with keys for physics loss computation:
                'predicted_A', 'predicted_Q', 'predicted_ql',
                't', 'x', 'hydraulic_radius', 'friction_slope'.

        Returns:
            Dictionary with 'total', 'data', 'physics_sv', 'physics_manning'.
        """
        data_loss = self.focal_loss(logits, targets)

        sv_loss = torch.tensor(0.0, device=logits.device)
        manning_loss = torch.tensor(0.0, device=logits.device)

        if physics_data is not None:
            if all(k in physics_data for k in ["predicted_A", "predicted_Q", "predicted_ql", "t", "x"]):
                sv_loss = self.sv_loss(
                    physics_data["predicted_A"],
                    physics_data["predicted_Q"],
                    physics_data["predicted_ql"],
                    physics_data["t"],
                    physics_data["x"],
                )

            if all(k in physics_data for k in ["predicted_Q", "predicted_A", "hydraulic_radius", "friction_slope"]):
                manning_loss = self.manning_loss(
                    physics_data["predicted_Q"],
                    physics_data["predicted_A"],
                    physics_data["hydraulic_radius"],
                    physics_data["friction_slope"],
                )

        total = data_loss + self.lambda_sv * sv_loss + self.lambda_manning * manning_loss

        return {
            "total": total,
            "data": data_loss,
            "physics_sv": sv_loss,
            "physics_manning": manning_loss,
        }
