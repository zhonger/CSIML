"""
Machine learning modules for small and imbalanced materials science datasets

SIML is a Python module based on scikit-learn and imbalanced-learn libraries as well as 
scientific Python packages (numpy, pandas, matplotlib, mpi4py, scikit-learn-intelex).

It aims to provide efficient and trustworthy solutions to learning and tackling the 
influence from the imbalance of materials science datasets.

See https://siml.readthedocs.org for complete documentation.
"""
import os

__version__ = "0.1.0"

# os.environ.setdefault("INTELEX", "False")

__all__ = [
    "Features"
]
