# HQC-PINN: Hybrid Quantum-Classical Physics-Informed Neural Networks for Hydrological Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-green.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **Variational Quantum Physics-Informed Neural Networks for Hydrological PDE-Constrained Learning with Inherent Uncertainty Quantification**
>
> Prasad Nimantha Madusanka Ukwatta Hewage
>
> *arXiv preprint* (2026) | Primary: `quant-ph` | Cross-list: `cs.LG`, `physics.geo-ph`

## Overview

This repository contains the complete implementation of the **Hybrid Quantum-Classical Physics-Informed Neural Network (HQC-PINN)** architecture for flood severity prediction using multi-modal satellite and meteorological data from the Kalu River basin, Sri Lanka.

### Key Contributions

1. **HQC-PINN Architecture**: A hybrid model integrating Variational Quantum Circuits (VQCs) into the Physics-Informed Neural Network framework, constrained by the Saint-Venant shallow water equations and Manning's flow equation.

2. **Quantum Uncertainty Quantification**: Leveraging Born-rule measurement stochasticity as a natural mechanism for uncertainty quantification without explicit Bayesian inference.

3. **Physics-Informed Trainability**: Theoretical and empirical analysis showing that hydrological physics constraints mitigate barren plateaus in variational quantum circuits.

4. **Quantum Transfer Learning**: A classical-to-quantum (CQ) transfer learning protocol pre-trained on multi-hazard disaster data (82 events) and fine-tuned for flood prediction.

5. **Parameter Efficiency**: Significant reduction in trainable parameters compared to equivalent classical PINNs while maintaining competitive accuracy, enabled by the VQC's exponentially large Hilbert space (2^8 = 256 dimensional).

### Architecture

```
Input (25-dim multi-modal features)
    │
    ▼
┌─────────────────────────┐
│  Classical Pre-Net       │  z = π · tanh(W₂ · ReLU(W₁·x + b₁) + b₂)
│  25 → 64 → 8            │  Learned compression to qubit register
└─────────┬───────────────┘
          │  8 values ∈ [-π, π]
          ▼
┌─────────────────────────┐
│  Variational Quantum     │  Angle Encoding (R_Y) → [R_Y, R_Z, CNOT]×3 layers
│  Circuit (8 qubits)      │  48 trainable parameters
│  ⟨Z₁⟩, ⟨Z₂⟩, ..., ⟨Z₈⟩  │  Parameter-shift rule gradients
└─────────┬───────────────┘
          │  8 expectation values ∈ [-1, 1]
          ▼
┌─────────────────────────┐
│  Classical Post-Net      │  (Linear → ReLU → Linear)
│  8 → 32 → 4             │
└─────────┬───────────────┘
          │
          ▼
    Output (4 flood severity classes)
```

## Data Sources

| Source | Type | Resolution | Features |
|--------|------|-----------|----------|
| Sentinel-1 SAR | Radar | 10m | VV, VH backscatter, ratios |
| Landsat 8/9 | Optical | 30m | NDVI, NDWI, MNDWI, EVI, BSI |
| ERA5-Land | Meteorological | 0.1° | Temperature, wind, pressure, soil moisture |
| CHIRPS | Precipitation | 0.05° | Daily and pentad rainfall |
| SRTM DEM | Topographic | 30m | Elevation, slope, TWI, curvature |
| NOAA STORM | Labels | Event-level | Flood severity classification |

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/HQC-PINN-Flood-Prediction.git
cd HQC-PINN-Flood-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Run the demo notebook

```bash
cd notebooks
jupyter notebook demo_hqc_pinn.ipynb
```

### Run the full experiment

```bash
# Quick demo (10 epochs, synthetic data)
python run_experiment.py --quick

# Full experiment
python run_experiment.py --config configs/default.json
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

## Project Structure

```
HQC-PINN-Flood-Prediction/
├── src/
│   ├── models/
│   │   ├── quantum_circuit.py    # VQC with angle encoding & HEA
│   │   ├── hqc_pinn.py          # Full HQC-PINN architecture
│   │   ├── classical_pinn.py    # Classical PINN baseline
│   │   └── quantum_transfer.py  # Quantum transfer learning
│   ├── training/
│   │   ├── physics_loss.py      # Saint-Venant & Manning losses
│   │   └── trainer.py           # Training loop with early stopping
│   ├── evaluation/
│   │   ├── metrics.py           # Classification & UQ metrics
│   │   └── ablation.py          # Ablation study framework
│   ├── data/
│   │   ├── dataset.py           # Flood & multi-hazard datasets
│   │   └── preprocessing.py     # Multi-modal feature extraction
│   └── utils/
│       ├── config.py            # Experiment configuration
│       └── visualization.py     # Publication-quality figures
├── configs/
│   └── default.json             # Default hyperparameters
├── notebooks/
│   └── demo_hqc_pinn.ipynb      # Interactive demonstration
├── tests/
│   ├── test_quantum_circuit.py  # VQC unit tests
│   ├── test_hqc_pinn.py        # Model integration tests
│   ├── test_physics_loss.py     # Physics loss tests
│   └── test_dataset.py          # Dataset tests
├── data/
│   ├── raw/                     # Raw satellite/meteo data
│   └── processed/               # Processed .npz datasets
├── results/
│   ├── figures/                 # Generated plots
│   └── logs/                    # Training logs
├── run_experiment.py            # Main experiment runner
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Key Results (from paper)

Results from the full-scale experiment on Kalu River basin data (see paper Tables I-V):

| Model | Accuracy (%) | F1 (macro) | Epochs to Target Loss |
|-------|-------------|------------|----------------------|
| **HQC-PINN (8q, 3L)** | **71.8** | **0.731** | **26** |
| cPINN | 69.4 | 0.705 | 94 |
| VQC-only (8q, 3L) | 67.8 | 0.682 | 51 |
| QTL (8q, 3L) | 73.6 | 0.742 | --- |

**Parameter comparison** (computed from code architecture):

| Component | cPINN | HQC-PINN (8q, 3L) |
|-----------|-------|-------------------|
| Classical layers | 48,068 | 2,604 |
| Quantum parameters (VQC) | 0 | 48 |
| **Total** | **48,068** | **2,652** |

> **Note:** Running `python run_experiment.py` on synthetic data will produce different accuracy
> numbers than the paper. The paper reports results from the full Kalu River basin dataset.

## Quantum Circuit Details

- **Qubits**: 8 (matching Kalu River basin feature compression)
- **Layers**: 3 (hardware-efficient ansatz)
- **Encoding**: Trainable angle encoding via R_Y gates
- **Ansatz**: Alternating R_Y/R_Z rotations + nearest-neighbor CNOT
- **Measurement**: Pauli-Z expectation values on all qubits
- **Gradient method**: Parameter-shift rule (hardware-compatible)
- **Simulator**: PennyLane `default.qubit` (compatible with IBM/IonQ hardware)

## Uncertainty Quantification

The quantum measurement process provides inherent uncertainty estimates:

```python
model = HQCPINN(input_dim=25, n_qubits=8, n_layers=3, n_classes=4)
results = model.predict_with_uncertainty(x, n_shots=200)

# results['probs_mean']             - Mean class probabilities
# results['entropy']                - Predictive entropy (total uncertainty)
# results['aleatoric_uncertainty']  - Aleatoric uncertainty from measurement noise
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hewage2026hqcpinn,
  title={Variational Quantum Physics-Informed Neural Networks for
         Hydrological PDE-Constrained Learning with
         Inherent Uncertainty Quantification},
  author={Hewage, Prasad Nimantha Madusanka Ukwatta},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  primaryClass={quant-ph}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- PennyLane (Xanadu) quantum ML framework
- Google Earth Engine and USGS for satellite data access
- Department of Irrigation, Sri Lanka for hydrological observations
- University of Bedfordshire for computational resources
