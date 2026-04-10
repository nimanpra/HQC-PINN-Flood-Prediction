"""
Dataset classes for flood prediction and multi-hazard data.

Handles multi-modal satellite and meteorological data from
the Kalu River basin, Sri Lanka (Section VII.A of paper).

Data sources:
  - Sentinel-1 SAR (VV/VH backscatter)
  - Landsat 8/9 (spectral indices: NDVI, NDWI, MNDWI)
  - ERA5-Land (temperature, wind, pressure, soil moisture)
  - CHIRPS (daily/pentad precipitation)
  - SRTM DEM (elevation, slope, aspect, TWI)
  - NOAA STORM Events Database (flood labels)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


FEATURE_GROUPS = {
    "sar": {
        "features": ["vv_backscatter", "vh_backscatter", "vv_vh_ratio", "cross_pol_diff"],
        "dim": 4,
    },
    "spectral": {
        "features": ["ndvi", "ndwi", "mndwi", "evi", "bsi"],
        "dim": 5,
    },
    "meteorological": {
        "features": [
            "temperature_2m", "dewpoint_2m", "u_wind_10m", "v_wind_10m",
            "surface_pressure", "soil_moisture_l1", "soil_moisture_l2",
            "total_evaporation",
        ],
        "dim": 8,
    },
    "precipitation": {
        "features": ["chirps_daily", "chirps_pentad", "era5_total_precip"],
        "dim": 3,
    },
    "topographic": {
        "features": ["elevation", "slope", "aspect", "twi", "curvature"],
        "dim": 5,
    },
}

TOTAL_FEATURES = sum(g["dim"] for g in FEATURE_GROUPS.values())

FLOOD_CLASSES = {
    0: "No Flood",
    1: "Low Severity",
    2: "Moderate Severity",
    3: "Severe Flood",
}

CLASS_DISTRIBUTION = {
    0: 0.91,
    1: 0.04,
    2: 0.04,
    3: 0.01,
}


class FloodDataset(Dataset):
    """Kalu River basin flood prediction dataset.

    Each sample contains multi-modal features from co-registered
    satellite and meteorological data, with flood severity labels
    derived from NOAA STORM Events and field surveys.

    Args:
        data_path: Path to processed .npz file.
        split: One of 'train', 'val', 'test'.
        include_physics: If True, include physics collocation point data.
    """

    def __init__(
        self,
        data_path: str = None,
        split: str = "train",
        include_physics: bool = True,
    ):
        self.split = split
        self.include_physics = include_physics
        self.features = None
        self.targets = None
        self.physics_data = None

        if data_path and Path(data_path).exists():
            data = np.load(data_path)
            self.features = torch.tensor(data[f"{split}_features"], dtype=torch.float32)
            self.targets = torch.tensor(data[f"{split}_targets"], dtype=torch.long)
            if include_physics and f"{split}_physics" in data:
                self.physics_data = {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in data[f"{split}_physics"].items()
                }

    @classmethod
    def generate_synthetic(
        cls, n_samples: int = 1000, seed: int = 42, include_physics: bool = True
    ) -> "FloodDataset":
        """Generate synthetic dataset for testing and demonstration.

        Creates physically plausible multi-modal features with
        realistic class imbalance matching the Kalu River basin data.
        """
        np.random.seed(seed)
        dataset = cls.__new__(cls)
        dataset.include_physics = include_physics

        features = np.random.randn(n_samples, TOTAL_FEATURES).astype(np.float32)

        # Approximate observed class imbalance
        class_probs = [0.91, 0.04, 0.04, 0.01]
        targets = np.random.choice(4, size=n_samples, p=class_probs)

        for i, label in enumerate(targets):
            if label > 0:
                features[i, 8] += label * 1.5   # Higher precip
                features[i, 0] -= label * 0.5    # Lower VV backscatter (water)
                features[i, 11] += label * 0.8   # Higher soil moisture

        dataset.features = torch.tensor(features, dtype=torch.float32)
        dataset.targets = torch.tensor(targets, dtype=torch.long)
        dataset.split = "synthetic"

        if include_physics:
            dataset.physics_data = {
                "t": torch.rand(n_samples, requires_grad=True),
                "x": torch.rand(n_samples, requires_grad=True),
                "predicted_A": torch.rand(n_samples) * 50 + 10,
                "predicted_Q": torch.rand(n_samples) * 100 + 5,
                "predicted_ql": torch.rand(n_samples) * 2,
                "hydraulic_radius": torch.rand(n_samples) * 3 + 0.5,
                "friction_slope": torch.rand(n_samples) * 0.01 + 0.001,
            }

        return dataset

    def __len__(self):
        return len(self.features) if self.features is not None else 0

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "targets": self.targets[idx],
        }
        if self.include_physics and self.physics_data is not None:
            item["physics_data"] = {
                k: v[idx] for k, v in self.physics_data.items()
            }
        return item

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for focal loss."""
        if self.targets is None:
            return torch.ones(4)
        counts = torch.bincount(self.targets, minlength=4).float()
        counts = counts.clamp(min=1)
        weights = 1.0 / counts
        return weights / weights.sum() * len(FLOOD_CLASSES)


class MultiHazardDataset(Dataset):
    """Multi-hazard disaster dataset for transfer learning (Section VI).

    Pre-training data from 82 disaster events across multiple
    hazard types (floods, cyclones, landslides, droughts).
    """

    def __init__(self, data_path: str = None, split: str = "train"):
        self.split = split

        if data_path and Path(data_path).exists():
            data = np.load(data_path)
            self.features = torch.tensor(data[f"{split}_features"], dtype=torch.float32)
            self.targets = torch.tensor(data[f"{split}_targets"], dtype=torch.long)
        else:
            self.features = None
            self.targets = None

    @classmethod
    def generate_synthetic(cls, n_samples: int = 5000, n_events: int = 82, seed: int = 42):
        """Generate synthetic multi-hazard data for demonstration."""
        np.random.seed(seed)
        dataset = cls.__new__(cls)

        features = np.random.randn(n_samples, TOTAL_FEATURES).astype(np.float32)
        hazard_types = np.random.choice(4, size=n_samples)  # 4 hazard types

        dataset.features = torch.tensor(features, dtype=torch.float32)
        dataset.targets = torch.tensor(hazard_types, dtype=torch.long)
        dataset.split = "synthetic"
        return dataset

    def __len__(self):
        return len(self.features) if self.features is not None else 0

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
        }
