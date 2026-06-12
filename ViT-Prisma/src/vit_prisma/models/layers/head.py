
import torch.nn as nn
import torch

from typing import Dict, Union

from jaxtyping import Float

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from fancy_einsum import einsum
import warnings

# maps to class logits
class Head(nn.Module):
    def __init__(
            self, 
            cfg: Union[Dict, HookedViTConfig],
            d_model: int = 768,
            n_classes: int = 1000,
            dtype: torch.dtype = torch.float32
            ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        self.cfg = cfg

        if (hasattr(cfg, "d_model") 
            and hasattr(cfg, "n_classes")
            and hasattr(cfg, "dtype")):
            self.W_H: Float[torch.Tensor, "d_model n_classes"] = nn.Parameter(
                torch.empty(self.cfg.d_model, self.cfg.n_classes, dtype=cfg.dtype)
            )
            self.b_H: Float[torch.Tensor, "n_classes"] = nn.Parameter(
                torch.zeros(self.cfg.n_classes, dtype=cfg.dtype)
            )
        else:
            warnings.warn(
                "Config object missing d_model, n_classes, or dtype. Using default values for head layer parameters."
            )
            self.W_H: Float[torch.Tensor, "d_model n_classes"] = nn.Parameter(
                torch.empty(d_model, n_classes, dtype=dtype)
            )
            self.b_H: Float[torch.Tensor, "n_classes"] = nn.Parameter(
                torch.zeros(n_classes, dtype=dtype)
            )

    def forward(
        self, residual: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch n_classes"]:
        
        return (
            einsum(
                "batch d_model, d_model n_classes -> batch n_classes",
                residual,
                self.W_H,
            )
            + self.b_H
        )