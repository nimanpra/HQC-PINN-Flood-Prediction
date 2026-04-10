"""Tests for dataset classes."""

import pytest
import torch
from src.data.dataset import FloodDataset, MultiHazardDataset, TOTAL_FEATURES


class TestFloodDataset:
    def test_synthetic_generation(self):
        dataset = FloodDataset.generate_synthetic(n_samples=100)
        assert len(dataset) == 100
        assert dataset.features.shape == (100, TOTAL_FEATURES)
        assert dataset.targets.shape == (100,)

    def test_class_imbalance(self):
        dataset = FloodDataset.generate_synthetic(n_samples=10000)
        counts = torch.bincount(dataset.targets, minlength=4)
        assert counts[0] > counts[1]
        assert counts[0] > counts[3]

    def test_getitem(self):
        dataset = FloodDataset.generate_synthetic(n_samples=50)
        item = dataset[0]
        assert "features" in item
        assert "targets" in item
        assert item["features"].shape == (TOTAL_FEATURES,)

    def test_class_weights(self):
        dataset = FloodDataset.generate_synthetic(n_samples=1000)
        weights = dataset.class_weights()
        assert weights.shape == (4,)
        assert weights[3] > weights[0]  # Rare class gets higher weight


class TestMultiHazardDataset:
    def test_synthetic_generation(self):
        dataset = MultiHazardDataset.generate_synthetic(n_samples=500)
        assert len(dataset) == 500
        assert dataset.features.shape == (500, TOTAL_FEATURES)
