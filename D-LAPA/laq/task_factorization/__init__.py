"""UniVLA-style language factorization for task-centric depth features."""

from .config import ModelConfig
from .model import DepthFactorizedLAM

__all__ = ["DepthFactorizedLAM", "ModelConfig"]
