import logging
import torch.nn as nn
import torch 
from jaxtyping import Float
from vit_prisma.configs import HookedViTConfig
import einops

class PatchEmbedding(nn.Module):

    def __init__(self, config, logger=None,n_channels:int = 3, d_model:int = 768,kernel_size:int = 16, stride:int = 16):
        super().__init__()
        self.logger = logger
        self.config = config
        self.n_channels = n_channels
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.stride = stride

        if hasattr(self.config, "n_channels") and hasattr(self.config, "d_model") and hasattr(self.config, "patch_size"):
            self.proj = nn.Conv2d(
                self.config.n_channels, 
                self.config.d_model, 
                kernel_size=self.config.patch_size, 
                stride=self.config.patch_size, 
                bias=True
            )
        elif all(param is not None for param in [n_channels, d_model, kernel_size, stride]):
            self.proj = nn.Conv2d(
            n_channels=self.n_channels, 
            d_model=self.d_model,
            kernel_size= self.kernel_size,
            stride=self.stride,
            bias=True,
        )
        else:
            raise ValueError("PatchEmbedding projection layer not initialized due to missing configuration parameters. Please ensure n_channels, d_model, and patch_size are specified in the configuration.")
    
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

    def __init__(self, cfg:HookedViTConfig, model_name:str, tublet_size :list = None, n_channels:int = 3, d_model:int = 768):
        super().__init__()
        self.cfg = cfg
        self.proj = None
        self.model_name = model_name
        self.tublet_size = tublet_size
        self.n_channels = n_channels
        self.d_model = d_model

        if (hasattr(self.cfg, "video_tubelet_depth") 
            and hasattr(self.cfg, "patch_size")
            ):
            
            tubelet_size = [self.cfg.video_tubelet_depth, self.cfg.patch_size, self.cfg.patch_size]

        elif tublet_size is not None:
            
            logging.warning("TubeletEmbedding projection layer not initialized due to missing configuration parameters.")
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
            self.proj = nn.Conv3d(
                n_channels=self.n_channels, 
                d_model=self.d_model,
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

