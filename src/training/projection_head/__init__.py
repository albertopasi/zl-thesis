"""
Projection Head Module

Reduces high-dimensional REVE features to lower-dimensional representations
for contrastive loss computation.
"""

from .projection_head import ProjectionHead
from .config import ProjectionHeadConfig

__all__ = [
    "ProjectionHead",
    "ProjectionHeadConfig",
]
