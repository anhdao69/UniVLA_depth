from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModelConfig:
    stage: str = "irr"
    image_size: int = 256
    patch_size: int = 32
    dim: int = 1024
    encoder_depth: int = 8
    decoder_depth: int = 8
    heads: int = 16
    mlp_ratio: int = 4
    dropout: float = 0.0
    num_queries: int = 4
    vq_dim: int = 32
    irr_codebook_size: int = 16
    task_codebook_size: int = 8
    text_dim: int = 768
    commitment_beta: float = 0.25

    def __post_init__(self) -> None:
        if self.stage not in {"irr", "task"}:
            raise ValueError(f"stage must be 'irr' or 'task', got {self.stage!r}")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.num_queries != 4:
            raise ValueError("the downstream contract requires exactly four queries")
        if self.vq_dim <= 0 or self.dim <= 0:
            raise ValueError("model and VQ dimensions must be positive")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any], *, stage: str | None = None) -> "ModelConfig":
        data = dict(value)
        if stage is not None:
            data["stage"] = stage
        return cls(**data)
