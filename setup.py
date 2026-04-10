from setuptools import setup, find_packages

setup(
    name="hqc-pinn-flood",
    version="1.0.0",
    description="Hybrid Quantum-Classical Physics-Informed Neural Networks for Hydrological Prediction",
    author="Prasad Nimantha Madusanka Ukwatta Hewage",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pennylane>=0.38.0",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
