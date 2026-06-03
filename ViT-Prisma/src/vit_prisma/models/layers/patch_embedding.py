import logging
import torch.nn as nn
import torch 
from jaxtyping import Float
from vit_prisma.configs import HookedViTConfig
import einops
import warnings
class PatchEmbedding(nn.Module):

    def __init__(self, config, logger=None,in_channels:int = 3,out_channels:int=2,kernel_size:int = 16, stride:int = 16):
        super().__init__()
        self.logger = logger
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if hasattr(self.config, "in_channels") and hasattr(self.config, "out_channels") and hasattr(self.config, "patch_size"):
            
            self.proj = nn.Conv2d(
                self.config.in_channels, 
                self.config.out_channels, 
                kernel_size=self.config.patch_size, 
                stride=self.config.patch_size, 
                bias=True
            )
        elif all(param is not None for param in [in_channels, out_channels, kernel_size, stride]):
            warnings.warn("Using configuration parameters for PatchEmbedding projection layer. Ensure that n_channels, out_channels, and patch_size are correctly specified in the configuration.")
            
            self.proj = nn.Conv2d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True
            )
        else:
            raise ValueError("PatchEmbedding projection layer not initialized due to missing configuration parameters. Please ensure n_channels, out_channels, and patch_size are specified in the configuration.")
    
    def _log(self, stage, tensor):
        if self.logger:
            self.logger.info(f"{stage} size: {tensor.shape}")

    def forward(self, x:Float[torch.Tensor, "batch channel height width"]) -> Float[torch.Tensor, "batch n_tokens d_model"]:
        self._log("PatchEmbedding input", x)

        x = self.proj(x).flatten(2).transpose(1, 2)

        self._log("PatchEmbedding output", x)
        return x
    

# Used for videos, 3 dimensional spacetime patches 
class TubeletEmbedding(nn.Module):

    def __init__(self, cfg:HookedViTConfig, model_name:str, tubelet_size :list = None, in_channels:int = 3, out_channels:int = 2):
        super().__init__()
        self.cfg = cfg
        self.proj = None
        self.model_name = model_name
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if (hasattr(self.cfg, "video_tubelet_depth") 
            and hasattr(self.cfg, "patch_size")
            ):
            
            tubelet_size = [self.cfg.video_tubelet_depth, self.cfg.patch_size, self.cfg.patch_size]

        elif self.tubelet_size is None:
            
            warnings.warn("TubeletEmbedding projection layer not initialized due to missing configuration parameters.")
            tubelet_size_2 = ["VideoMAEv2-Base","VideoMAEv2-Large","vjepa","vjepa-large","vivit-large","vivit-huge"]
            if any(name in model_name for name in tubelet_size_2):
                tubelet_size = [2, 16, 16]  # Default tubelet size for these models
        
        else:
            
            raise ValueError("TubeletEmbedding projection layer not initialized due to missing configuration parameters and tubelet size not provided.Please manually specify the tubelet size for the model.")

        if hasattr(self.cfg, "n_channels") and hasattr(self.cfg, "d_model"):    
            self.proj = nn.Conv3d(
                self.cfg.n_channels, 
                self.cfg.d_model, 
                kernel_size=tubelet_size, 
                stride=tubelet_size, 
                bias=True
            )
        else:
            warnings.warn("TubeletEmbedding projection layer initialized with default in_channels and out_channels. Ensure that n_channels and d_model are correctly specified in the configuration for optimal performance.")
            self.proj = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=tubelet_size,
                stride=tubelet_size,
                bias=True,
            )

                               
                
    def forward(self, x:Float[torch.Tensor, "batch num_frames channels height width"]) -> Float[torch.Tensor, "batch n_tokens d_model"]:
        if self.proj is None:
            raise ValueError("TubeletEmbedding projection layer not initialized due to missing configuration parameters.")
        
        # Flip num_frames and channels
        # x = einops.rearrange(x, "b t c h w -> b c t h w")
        
        x = self.proj(x)

        # Flatten the tokens
        x = einops.rearrange(x, "b c t h w -> b (t h w) c") 

        return x

