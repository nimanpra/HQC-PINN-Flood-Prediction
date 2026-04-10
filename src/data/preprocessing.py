"""
Multi-modal data preprocessing pipeline.

Handles satellite imagery, meteorological data, and topographic
feature extraction and normalization for the Kalu River basin study.
"""

import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class NormalizationStats:
    """Per-feature normalization statistics."""
    mean: np.ndarray = field(default_factory=lambda: np.zeros(25))
    std: np.ndarray = field(default_factory=lambda: np.ones(25))


class MultiModalPreprocessor:
    """Preprocessing pipeline for multi-modal flood prediction data.

    Handles feature extraction, normalization, and fusion from
    heterogeneous data sources:
      - SAR: dB conversion, speckle filtering, ratio computation
      - Spectral: Index computation, cloud masking
      - Meteorological: Temporal aggregation, anomaly detection
      - Topographic: Terrain derivative computation

    Args:
        feature_dim: Expected total feature dimension (25 for full multi-modal).
    """

    def __init__(self, feature_dim: int = 25):
        self.feature_dim = feature_dim
        self.stats = NormalizationStats()
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> "MultiModalPreprocessor":
        """Compute normalization statistics from training data.

        Args:
            features: Training features of shape (n_samples, feature_dim).
        """
        self.stats.mean = features.mean(axis=0)
        self.stats.std = features.std(axis=0)
        self.stats.std[self.stats.std < 1e-8] = 1.0
        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            features: Raw features of shape (n_samples, feature_dim).

        Returns:
            Normalized features.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return (features - self.stats.mean) / self.stats.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    @staticmethod
    def compute_sar_features(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """Compute SAR-derived features from Sentinel-1 data.

        Args:
            vv: VV polarization backscatter (dB).
            vh: VH polarization backscatter (dB).

        Returns:
            Array of [VV, VH, VV/VH ratio, cross-pol difference].
        """
        vv_db = 10 * np.log10(np.clip(vv, 1e-10, None))
        vh_db = 10 * np.log10(np.clip(vh, 1e-10, None))
        ratio = vv_db - vh_db
        diff = vv_db + vh_db
        return np.stack([vv_db, vh_db, ratio, diff], axis=-1)

    @staticmethod
    def compute_spectral_indices(
        nir: np.ndarray,
        red: np.ndarray,
        green: np.ndarray,
        swir: np.ndarray,
        blue: np.ndarray,
    ) -> np.ndarray:
        """Compute spectral indices from Landsat bands.

        Returns:
            Array of [NDVI, NDWI, MNDWI, EVI, BSI].
        """
        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)
        mndwi = (green - swir) / (green + swir + eps)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
        bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)
        return np.stack([ndvi, ndwi, mndwi, evi, bsi], axis=-1)

    @staticmethod
    def compute_twi(slope: np.ndarray, contributing_area: np.ndarray) -> np.ndarray:
        """Compute Topographic Wetness Index from SRTM DEM.

        TWI = ln(a / tan(beta))
        """
        slope_rad = np.radians(np.clip(slope, 0.001, None))
        return np.log(contributing_area / np.tan(slope_rad) + 1e-8)
