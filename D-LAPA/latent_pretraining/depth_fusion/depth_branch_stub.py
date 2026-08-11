from typing import Optional, Protocol

import numpy as np


class DepthBranch(Protocol):
    """Online depth branch contract to be implemented by the depth-model owner."""

    def encode(
        self,
        rgb_image: np.ndarray,
        instruction: Optional[str] = None,
        latent_action_4096: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a float32 vector with shape (1024,)."""


class UnimplementedDepthBranch:
    """Placeholder used until the online depth model is available."""

    def encode(
        self,
        rgb_image: np.ndarray,
        instruction: Optional[str] = None,
        latent_action_4096: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Online depth branch is intentionally blank. Implement encode(...) "
            "to return a float32 depth feature with shape (1024,)."
        )
