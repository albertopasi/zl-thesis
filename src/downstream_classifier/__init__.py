"""
Downstream Classifier Module

Classification head for workload prediction and other downstream tasks
using REVE extracted features.
"""

from .classifier import DownstreamClassifier
from .config import ClassifierConfig

__all__ = [
    "DownstreamClassifier",
    "ClassifierConfig",
]
