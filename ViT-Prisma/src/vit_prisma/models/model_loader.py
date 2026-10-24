"""
Efficient Prisma Model Loading System
===================================

Configs and weights load directly from Huggingface, with config and weight conversion scripts
to make them compatible with the Prisma library.

"""

from ast import Load
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch

import torch.nn as nn
from vit_prisma.configs.HookedTextTransformerConfig import HookedTextTransformerConfig
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.sae import config
from vit_prisma.vjepa_hf.configs.configs import CONFIGS
from transformers import (AutoConfig, 
                          ViTModel, 
                          CLIPModel, 
                          VivitForVideoClassification, 
                          CLIPVisionModelWithProjection)
import json 
import os
import open_clip

# Import configuration dictionaries - these are just static data
from vit_prisma.models.model_config_registry import (
    MODEL_CATEGORIES,
    MODEL_CONFIGS,
    ModelCategory,
    TEXT_ONLY_SUPPORTED_MODELS,
)
from timm import create_model as timm_create_model

# Import conversion functions for reference only - will be loaded on demand
from vit_prisma.models.weight_conversion import (
    convert_clip_weights,
    convert_dino_weights,
    convert_kandinsky_clip_weights,
    convert_open_clip_text_weights,
    convert_open_clip_weights,
    convert_timm_weights,
    convert_vivet_weights,
    convert_vjepa_weights,
    download_pretrained_from_hf,
    load_state_dict,
    remove_open_clip_prefix,
)
from vit_prisma.utils.enums import ModelType
from abc import ABC, abstractmethod

# Type alias
ConfigType = Union[HookedViTConfig, HookedTextTransformerConfig]


DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


MODELS_MISSING_CONFIG = {
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": (
        "xlm-roberta-base-ViT-B-32",
        "laion5b_s13b_b90k",
    ),
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k": (
        "roberta-ViT-B-32",
        "laion2b_s12b_b32k",
    ),
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k": (
        "xlm-roberta-large-ViT-H-14",
        "frozen_laion5b_s13b_b90k",
    ),
    "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k": (
        "coca_ViT-B-32",
        "laion2b_s13b_b90k",
    ),
    "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k": (
        "coca_ViT-L-14",
        "laion2b_s13b_b90k",
    ),
}

# Models that pass and fail according to 'tests/test_loading_clip.py'. Please update list as models get fixed / break.
PASSING_MODELS = {
    "TINY_CLIP": {"wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"},
    "OPEN_CLIP": {
        "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K",
        "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
        "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
        "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.laion2b_e16",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32",
    "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",},
    "DINO": {
    "facebook/dino-vitb16",
    "facebook/dino-vitb8",},
    "VIT": {
    "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",}
}

FAILING_MODELS = {
    # # MODELS THAT FAIL CURRENTLY
    "OPEN_CLIP": {
    "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch16_clip_224.openai",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch32_clip_224.openai",
    "open-clip:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K",
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e31",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e32",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_400m",
    "open-clip:timm/vit_large_patch14_clip_224.openai",
    "open-clip:timm/vit_large_patch14_clip_336.openai",
    "open-clip:timm/vit_medium_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m",
    "open-clip:timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_huge_patch14_clip_224.metaclip_2pt5b",},
    "DINO": {
    "facebook/dino-vits16",
    "facebook/dino-vits8",
    }
}


# ===============================
# Core Model Loading Functions
# ===============================


def check_model_name_supported(
    model_name: str,
    model_type: ModelType,
) -> str|Tuple[str, ModelCategory]:
    """
    Check if a model name is valid and supported.
    Args:
        model_name: Name of the model to check
        model_type: Type of the model
    Returns:
        model_name: Potentially modified model name
        category: ModelCategory of the model
    >>> Example:
        >>> check_model_name_supported("open-clip:timm/vit_base_patch16_clip_224.openai", ModelType.VISION)
        ("open-clip:timm/vit_base_patch16_clip_224.openai", ModelCategory.OPEN_CLIP)
    """
    if model_name not in MODEL_CATEGORIES:
        raise ValueError(f"Model '{model_name}' is not registered in configurations")

    if isinstance(model_type, ModelType):
        if (model_type == ModelType.TEXT 
        and model_name not in TEXT_ONLY_SUPPORTED_MODELS):
            raise ValueError(f"Model '{model_name}' does not support text modality")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    if isinstance(model_name, str):
        category = MODEL_CATEGORIES[model_name]
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model_name, category

def get_dynamic_config(
    category: ModelCategory,
    model_name: str,
    model_type: ModelType,
    local_path: Optional[str] = None,
) -> ConfigType:
    '''
    Get the dynamic configuration for a given model name and type.
    Args:
        category: The category of the model
        model_name: Name of the model
        model_type: Type of the model
    Returns:
        The dynamic configuration object
    >>> Example:
        >>> get_dynamic_config("open-clip:timm/vit_base_patch16_clip_224.openai", ModelType.VISION)
        HookedViTConfig(...)
    '''
    # Get dynamic config from source
    if category == ModelCategory.TIMM:
        old_config = _get_timm_hf_config(model_name)
        new_config = _create_config_from_hf(old_config, model_name, model_type)
    elif category == ModelCategory.OPEN_CLIP:
        old_config = _get_open_clip_config(
            model_name, model_type, local_path=local_path
        )
        new_config = _create_config_from_open_clip(old_config, model_name, model_type)
    elif category == ModelCategory.DINO:
        old_config = _get_general_hf_config(model_name, model_type=None)
        new_config = _create_config_from_hf(old_config, model_name, model_type=None)
    elif category in [ModelCategory.CLIP, ModelCategory.VIVIT]:
        old_config = _get_general_hf_config(model_name, model_type)
        new_config = _create_config_from_hf(old_config, model_name, model_type)
    elif category == ModelCategory.VJEPA:
        for model_name in CONFIGS:
            if model_name == model_name:
                new_config = CONFIGS[model_name]
                old_config = None  # No HuggingFace config for VJEPA models, just use our custom config

    else:
        raise ValueError(f"Unsupported model category: {category} for model '{model_name}'")

    return new_config

def apply_registry_overrides( model_name, 
                                 model_type) -> ConfigType:
    """
    Apply registry overrides to the configuration.
    Args:
        new_config: The configuration object to modify
        model_name: Name of the model
        model_type: Type of the model (VISION or TEXT)
    Returns:
        The modified configuration object
    >>> Example:
        >>> apply_registry_overrides(new_config, "open-clip:timm/vit_base_patch16_clip_224.openai", ModelType.VISION)
        HookedViTConfig(...)
        
    """
    # Apply registry overrides
    new_config = get_dynamic_config(model_name, model_type)
    registry_overrides = MODEL_CONFIGS[model_type].get(model_name, {})
    for key, value in registry_overrides.items():
        setattr(new_config, key, value)

    if hasattr(new_config, "n_heads") and hasattr(new_config, "d_model"):
        new_config.d_head = (
            new_config.d_model // new_config.n_heads
        )  # Calculate this after retrieving latest info
    else:
        new_config.d_head = 64  # Default to 64-dim heads if not specified
        logging.warning(
                f"Model '{model_name}' does not have explicit head dimensions. Using default."
            )
    return new_config


def check_model_name(model_name: str, 
                     allow_failing: bool = False) -> str:
    """
    Check if a model name is valid and supported.

    Args:
        model_name: Name of the model to check
        allow_failing: Whether to allow loading models that are known to fail tests

    Returns:
        Potentially modified model name
    >>> Example:
        >>> check_model_name("open-clip:timm/vit_base_patch16_clip_224.openai")
        "open-clip:timm/vit_base_patch16_clip_224.openai"
    """
    if model_name in MODELS_MISSING_CONFIG:
        model_name = MODELS_MISSING_CONFIG[model_name][0]
        logging.warning(
            f"Model '{model_name}' is missing a configuration in the registry. Using '{model_name}' instead."
        )

    if model_name in FAILING_MODELS:
        msg = f"Model '{model_name}' is in the list of models failing tests."
        if not allow_failing:
            raise ValueError(msg + " Set allow_failing=True to load anyway.")
        else:
            logging.warning(msg + " Loading anyway as allow_failing=True.")
    elif model_name in PASSING_MODELS:
        logging.info(f"Model '{model_name}' is supported and passes tests.")
    else:
        logging.warning(
            f"Model '{model_name}' is not in the lists of models passing or failing tests. Unclear status. You may want to check that the HookedViT matches the original model under tests/test_loading_clip.py."
        )

    return model_name


#def load_weights(
#    model: nn.Module,
#    model_name: str,
#    model_type: ModelType,
#    dtype: torch.dtype,
#    **kwargs,
#) -> None:
#    """
#    Load and apply weights to a model.

#    Args:
#        model: Model to load weights into
#        model_name: Name of the model
#        model_type: Type of model (VISION or TEXT)
#        dtype: Data type for weights
#        **kwargs: Additional arguments
#    """
#    category = MODEL_CATEGORIES[model_name]
#    config = model.cfg
    # Load and convert weights
#    original_weights = load_original_weights(
#        model_name, category, model_type, dtype, **kwargs
#    )
#    converted_weights = convert_weights(
#        original_weights, model_name, category, config, model_type
#    )

    # Apply weights to model
#    full_state_dict = fill_missing_keys(model, converted_weights)

#    return full_state_dict



#TODO: write a function to create a dictionary of weight dictionaries

     
def convert_weights(
    original_weights: Any,
    model_name: str,
    category: ModelCategory,
    config: ConfigType,
    model_type: ModelType,
) -> Dict[str, torch.Tensor]:
    """
    Convert weights for a specific model into Prisma-compatible format.
    >>> Example:
        >>> convert_weights(original_weights, "open-clip:timm/vit_base_patch16_clip_224.openai", ModelCategory.OPEN_CLIP, config, ModelType.VISION)
        {'weight_name': tensor(...), ...}
    """
    if (
        "eva02" in model_name.lower() or "eva_" in model_name.lower()
    ) and category == ModelCategory.OPEN_CLIP:
        return convert_timm_weights(original_weights, config)

    if category == ModelCategory.CLIP and model_type == ModelType.VISION:
        vision_weights = original_weights.vision_model.state_dict()
        projection_weights = original_weights.visual_projection.state_dict()
        return convert_clip_weights(vision_weights, projection_weights, config)

    if category == ModelCategory.TIMM:
        converter = convert_timm_weights
    elif category == ModelCategory.OPEN_CLIP:
        converter = (
            convert_open_clip_text_weights
            if model_type == ModelType.TEXT
            else convert_open_clip_weights
        )
    elif category == ModelCategory.DINO:
        converter = convert_dino_weights
    elif category == ModelCategory.VIVIT:
        converter = convert_vivet_weights
    elif category == ModelCategory.VJEPA:
        converter = convert_vjepa_weights
    elif category == ModelCategory.KANDINSKY:
        converter = convert_kandinsky_clip_weights
    else:
        raise ValueError(f"No converter available for {category} with {model_type}")

    return converter(original_weights, config)


#def load_weights(
#    model: nn.Module,
#    model_name: str,
#    model_type: ModelType,
#    dtype: torch.dtype,
#    **kwargs,
#) -> Dict[str, torch.Tensor]:
#    """Load and convert weights into a Prisma-compatible state dict."""
#    category = MODEL_CATEGORIES[model_name]
#    config = model.cfg

#    original_weights = load_original_weights(
#        model_name, category, model_type, dtype, **kwargs
#    )
#    converted_weights = convert_weights(
#        original_weights, model_name, category, config, model_type
#    )

#    return fill_missing_keys(model, converted_weights)

class ModelLoader(ABC):
    """
    Abstract class to load models with configuration and weights.
    """
    @abstractmethod
    def check_model_name(self):
        """
        Check if the model name is valid and supported.
        Raises:
            ValueError: If the model name is not valid or supported
        """
        pass

    @abstractmethod
    def _get_general_hf_config(self):
        """
        Get HuggingFace config from TIMM model.
        Returns:
            hf_config: The HuggingFace configuration object
        """
        pass

    @abstractmethod
    def _get_timm_hf_config(self):
        """
        Get HuggingFace config from TIMM model.
        Returns:
            hf_config: The HuggingFace configuration object
        """
        pass

    @abstractmethod
    def _get_open_clip_config(self):
        """
        Get HuggingFace config from OpenCLIP model.
        Returns:
            hf_config: The HuggingFace configuration object
        """
        pass

    @abstractmethod
    def _create_clip_config_from_hf(self):
        """
        Create a Prisma-compatible config from HuggingFace config.
        Returns:
            new_config: The Prisma-compatible configuration object
        """
        pass

    @abstractmethod
    def _create_config_from_hf(self):
        """
        Create a Prisma-compatible config from HuggingFace config.
        Returns:
            new_config: The Prisma-compatible configuration object
        """
        pass

    @abstractmethod
    def _create_config_from_open_clip(self):
        """
        Create a Prisma-compatible config from OpenCLIP config.
        Returns:
            new_config: The Prisma-compatible configuration object
        """
        pass

    @abstractmethod
    def create_config_object(self):
        """
        Create a configuration object for the model.
        Returns:
            config: The configuration object
        """
        pass

    @abstractmethod
    def fill_missing_keys(self):
        """
        Fill missing keys in the state dictionary with default values.
        Returns:
            full_state_dict: The complete state dictionary
        """
        pass

    @abstractmethod
    def apply_registry_overrides(self):
        """
        Apply registry overrides to the configuration.
        Returns:
            config: The modified configuration object
        """
        pass

    @abstractmethod
    def check_model_name_supported(self):
        """
        Check if a model name is valid and supported.
        Raises:
            ValueError: If the model name is not valid or supported
        """
        pass

    @abstractmethod
    def convert_weights(self):
        """
        Convert weights for a specific model into Prisma-compatible format.
        Returns:
            converted_weights: The converted weights
        """
        pass

    
    @abstractmethod    
    def check_kwargs_for_quantization(self):
        """
        Check if quantization flags are set in kwargs and raise an error if so.
        Raises:
            AssertionError: If quantization flags are set
        """
        pass

    @abstractmethod    
    def convert_dtype_to_torch_dtype(self):
        """
        Check if dtype is a string and convert to torch dtype if so.
        Returns:
            dtype: The converted torch dtype
        """
        pass

    @abstractmethod
    def check_float16_on_cpu(self):
        """
        Check if float16 is being used on CPU and issue a warning if so.
        """
        pass

    @abstractmethod
    def override_config_with_kwargs(self):
        """
        Override configuration parameters with values from kwargs.
        Returns:
            config: The overridden configuration object
        """
        pass

    @abstractmethod
    def instantiate_model_class(self):
        """
        Instantiate the model class.
        Returns:
            An instance of the model class
        """
        pass

    @abstractmethod
    def initialise_model(self):
        """
        Initialise the model with configuration and weights.
        Returns:
            Loaded model
        """
        pass

    @abstractmethod
    def load_weights(self):
        """
        Load and apply weights to the model.
        Returns:
            full_state_dict: The complete state dictionary
        """
        pass


    

#    def load_hooked_model(
#        self,
#        model_name: str,
#    model_class: Type = None,
#    model_type: ModelType = ModelType.VISION,
#    device: str = "cuda",
#    dtype: torch.dtype = torch.float32,
#    pretrained: bool = True,
#    fold_ln: bool = False,
#    center_writing_weights: bool = False,
#    fold_value_biases: bool = True,
#    refactor_factored_attn_matrices: bool = False,
#    move_to_device: bool = True,
#    use_attn_result: bool = False,
#    allow_failing: bool = False,
#    local_path: Optional[str] = None,
#    config_overrides: Optional[Dict[str, Any]] = None,
#    **kwargs,
#) -> Any:
#    """
#    Load a model with configuration and weights.

#    Args:
#        model_name: Name of the model to load
#        model_class: Class to instantiate (optional, will be inferred if None)
#        model_type: Type of model (VISION or TEXT)
#        device: Device to load model on
#        dtype: Data type for model parameters
#        pretrained: Whether to load pretrained weights
#        fold_ln: Whether to fold layer normalization into attention
#        center_writing_weights: Whether to center writing weights
#        allow_failing: Whether to allow loading models that are known to fail tests
#        local_path: Path to local model weights (for OpenCLIP models)
#        **kwargs: Additional arguments

#    Returns:
#        Loaded model
#    """
class HookedModelLoader(ModelLoader):
    """
    Class to load models with configuration and weights.
    """
    def __init__(self, 
                 model_name: str,
                 model_class: Type = None,
                 model_type: ModelType = ModelType.VISION,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32,
                 pretrained: bool = True,
                 fold_ln: bool = False,
                 center_writing_weights: bool = False,
                 fold_value_biases: bool = True,
                 refactor_factored_attn_matrices: bool = False,
                 move_to_device: bool = True,
                 use_attn_result: bool = False,
                 allow_failing: bool = False,
                 local_path: Optional[str] = None,
                 config_overrides: Optional[Dict[str, Any]] = None,
                 hf_config: Optional[Dict[str, Any]] = None,
                 state_dict: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.pretrained = pretrained
        self.fold_ln = fold_ln
        self.center_writing_weights = center_writing_weights
        self.fold_value_biases = fold_value_biases
        self.refactor_factored_attn_matrices = refactor_factored_attn_matrices
        self.move_to_device = move_to_device
        self.use_attn_result = use_attn_result
        self.allow_failing = allow_failing
        self.local_path = local_path
        self.config_overrides = config_overrides or {}
        self.hf_config = hf_config
        self.state_dict = state_dict
        self.kwargs = kwargs

    def check_model_name(self):
        """
        Check if the model name is valid and supported.
        Raises:
            ValueError: If the model name is not valid or supported
        Returns:
            None
        >>> Example:
            >>> check_model_name("open-clip:timm/vit_base_patch16_clip_224.openai")
            None
        
        """
        if self.model_name in MODELS_MISSING_CONFIG:
                self.model_name = MODELS_MISSING_CONFIG[self.model_name][0]
                logging.warning(
                    f"Model '{self.model_name}' is missing a configuration in the registry. Using '{self.model_name}' instead."
                )
        
        if self.model_name in FAILING_MODELS:
            msg = f"Model '{self.model_name}' is in the list of models failing tests."
            if not self.allow_failing:
                raise ValueError(msg + " Set allow_failing=True to load anyway.")
            else:
                logging.warning(msg + " Loading anyway as allow_failing=True.")
        elif self.model_name in PASSING_MODELS:
            logging.info(f"Model '{self.model_name}' is supported and passes tests.")
        else:
            logging.warning(
                    f"Model '{self.model_name}' is not in the lists of models passing or failing tests. Unclear status. You may want to check that the HookedViT matches the original model under tests/test_loading_clip.py."
            )

    
    def _get_general_hf_config(self):  
        """
        Get HuggingFace config from TIMM model.
        Args:
            model_name: Name of the model to get config for
        Returns:
            hf_config: The HuggingFace configuration object
            >>> Example:
                >>> _get_general_hf_config("openai/clip-vit-large-patch14-336")
                AutoConfig(...)
        """
        if (self.model_type
                and isinstance(self.model_type, ModelType)):
                if self.model_type == ModelType.VISION:
                    self.model_type = "vision_config"
                elif self.model_type == ModelType.TEXT:
                    self.model_type = "text_config"
                hf_config = AutoConfig.from_pretrained(self.model_name)

        if (self.model_type
                and isinstance(self.model_type, ModelType)):
                hf_config = getattr(hf_config, self.model_type)
        return hf_config

    def _get_timm_hf_config(self):
        """
        Get HuggingFace config from TIMM model.
        Args:
            model_name: Name of the model to get config for
        Returns:
            hf_config: The HuggingFace configuration object
        >>> Example:
            >>> _get_timm_hf_config("timm/vit_base_patch16_clip_224.openai")
            AutoConfig(...)
        """
        timm_model = timm_create_model(self.model_name)
        hf_config = timm_model.pretrained_cfg
        return hf_config

    def _get_open_clip_config(self):
        '''
        Get HuggingFace config from OpenCLIP model.
        Args:
            model_name: Name of the model to get config for
            model_type: Type of the model (VISION or TEXT)
            local_path: Path to local model weights (for OpenCLIP models)
        Returns:
            hf_config: The HuggingFace configuration object
        >>> Example:
            >>> _get_open_clip_config("open-clip:timm/vit_base_patch16_clip_224.openai", ModelType.VISION)
            AutoConfig(...)
        '''
        if self.local_path is not None:
                # Look for config file in the local directory
                if os.path.isdir(self.local_path):
                    config_path = os.path.join(self.local_path, "open_clip_config.json")
                else:
                    # Assume local_path is the directory containing the model files
                    config_path = os.path.join(
                        os.path.dirname(self.local_path), "open_clip_config.json"
                    )
        
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        f"Config file not found at {config_path}. "
                        f"Please ensure 'open_clip_config.json' is in the same directory as your model weights."
                    )
        else:
            # Download from HuggingFace
            config_path = download_pretrained_from_hf(
                    remove_open_clip_prefix(self.model_name), filename="open_clip_config.json"
                )
        
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                model_config = config["model_cfg"]
        return model_config

    def _create_clip_config_from_hf(self):
        """
        Create a Prisma-compatible config from HuggingFace config.
        Args:
            hf_config: HuggingFace configuration object
        Returns:
            new_config: The Prisma-compatible configuration object
        >>> Example:
            >>> _create_clip_config_from_hf(hf_config)
            HookedViTConfig(...)
        """
        if self.model_type == ModelType.VISION:
            config = HookedViTConfig()

            # Extract patch size
            if hasattr(self.hf_config, "patch_size"):
                config.patch_size = self.hf_config.patch_size
            elif hasattr(self.hf_config, "tubelet_size"):
                config.patch_size = self.hf_config.tubelet_size[1]

    #         # Common attributes
            config.d_model = self.hf_config.vision_config.hidden_size
            config.n_layers = self.hf_config.vision_config.num_hidden_layers
            config.n_heads = self.hf_config.vision_config.num_attention_heads
            config.d_head = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_attention_heads
            config.d_mlp = self.hf_config.vision_config.intermediate_size
            config.image_size = getattr(self.hf_config, "image_size", 224)
            config.n_channels = getattr(self.hf_config, "num_channels", 3)
            config.patch_size = getattr(self.hf_config, "patch_size", 16)
            config.initializer_range = getattr(self.hf_config, "initializer_range", 0.02)

            config.model_name = self.model_name

            if hasattr(self.hf_config, "layer_norm_eps"):
                config.eps = self.hf_config.layer_norm_eps


            # Set output dimension appropriately
            if hasattr(self.hf_config, "projection_dim"):
                config.n_classes = self.hf_config.projection_dim
                config.return_type = "class_logits"
            elif hasattr(self.hf_config, "num_classes"):
                config.n_classes = self.hf_config.num_classes
                config.return_type = "class_logits"
            else:
                config.n_classes = config.d_model
                config.return_type = "pre_logits"


    #         # Video-specific settings
            if hasattr(self.hf_config, "tubelet_size"):
                config.is_video_transformer = True
                config.video_tubelet_depth = self.hf_config.tubelet_size[0]
                config.video_num_frames = self.hf_config.video_size[0]

        else:  # TEXT
            config = HookedTextTransformerConfig()
            config.d_model = self.hf_config.hidden_size
            config.n_layers = self.hf_config.num_hidden_layers
            config.n_heads = self.hf_config.num_attention_heads
            config.d_head = self.hf_config.hidden_size // self.hf_config.num_attention_heads
            config.d_mlp = self.hf_config.intermediate_size
            config.vocab_size = self.hf_config.vocab_size
            config.context_length = getattr(self.hf_config, "max_position_embeddings", 77)
            config.eps = self.hf_config.layer_norm_eps

        return config

    def _create_config_from_open_clip(self):
        """
        Create a Prisma-compatible config from OpenCLIP config.
        Args:
            hf_config: OpenCLIP configuration object
        Returns:
            new_config: The Prisma-compatible configuration object
        >>> Example:
            >>> _create_config_from_open_clip(hf_config)
            HookedViTConfig(...)
        """
        cfg = HookedViTConfig()
        if hasattr(self.model_cfg["vision_cfg"], "width"):
            cfg.d_model = self.model_cfg["vision_cfg"]["width"]
        else:
            cfg.d_model = 768  # Default to 768 if not specified
        if hasattr(self.model_cfg["vision_cfg"], "layers"):
            cfg.n_layers = self.model_cfg["vision_cfg"]["layers"]
        else:
            cfg.n_layers = 1  # Default to 1 if not specified
        if hasattr(self.model_cfg["vision_cfg"],"patch_size"):
            cfg.patch_size = self.model_cfg["vision_cfg"]["patch_size"]
        else:
            cfg.patch_size = 16  # Default to 16 if not specified
        if hasattr(self.model_cfg["vision_cfg"], "image_size"):
            cfg.image_size = self.model_cfg["vision_cfg"]["image_size"]
        else:
            cfg.image_size = 224  # Default to 224 if not specified
        if hasattr(self.model_cfg,"embed_dim"):
            cfg.n_classes = self.model_cfg["embed_dim"]
        else:
            cfg.n_classes = cfg.d_model  # Default to d_model if not specified
        
        
        cfg.model_name = self.model_name
        
        # Attention head number is not in open clip config, so we add it manually
        if "plus_clip" in self.model_name:
            cfg.n_heads = 14
        elif any(s in self.model_name for s in ["vit_xsmall"]):
            cfg.n_heads = 8
        elif any(s in self.model_name for s in ["ViT-B", "vit-base"]):
            cfg.n_heads = 12
        elif any(s in self.model_name for s in ["ViT-L", "vit_large", "vit_medium", "bigG"]):
            cfg.n_heads = 16
        elif any(s in self.model_name for s in ["huge_", "ViT-H"]):
            cfg.n_heads = 20
        elif any(s in self.model_name for s in ["ViT-g", "giant_"]):
            cfg.n_heads = 22
        elif any(s in self.model_name for s in ["gigantic_"]):
            cfg.n_heads = 26
        else:
            cfg.n_heads = 12
        
        # Set MLP dimension
        if self.model_cfg["vision_cfg"].get("mlp_ratio"):
            cfg.d_mlp = int(cfg.d_model * self.model_cfg["vision_cfg"].get("mlp_ratio"))
        else:
            cfg.d_mlp = cfg.d_model * 4
        
        # Common configurations
        cfg.normalization_type = "LN"
        cfg.return_type = "class_logits"
        
        return cfg

    def check_kwargs_for_quantization(self) -> None:
        """
        Check if quantization flags are set in kwargs and raise an error if so.
        Args:
            kwargs: Keyword arguments to check
        Raises:
            AssertionError: If quantization flags are set
        >>> Example:
            >>> check_kwargs_for_quantization({"load_in_8bit": True})
            AssertionError: Quantization not supported
        """
        assert not (
            self.kwargs.get("load_in_8bit", False) 
            or self.kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

    def convert_dtype_to_torch_dtype(self) -> torch.dtype:
        """
        Check if dtype is a string and convert to torch dtype if so.
        Args:
            dtype: Data type to check
        Returns:
            dtype: The converted torch dtype
        >>> Example:
            >>> convert_dtype_to_torch_dtype("float16")
            torch.float16
        """
        if isinstance(self.dtype, str):
            # Convert from string to a torch dtype
            dtype = DTYPE_FROM_STRING[self.dtype]
        return dtype

    def check_float16_on_cpu(self) -> None:
        """
        Check if float16 is being used on CPU and issue a warning if so.
        Args:
            dtype: Data type to check
            device: Device to check
            kwargs: Keyword arguments
        >>> Example:
            >>> check_float16_on_cpu(torch.float16, "cpu", {})
            None
        """
        if "torch_dtype" in self.kwargs:
            # For backwards compatibility with the previous way to do low precision loading
            # This should maybe check the user did not explicitly set dtype *and* torch_dtype
            dtype = self.kwargs["torch_dtype"]

        if (
            (self.kwargs.get("torch_dtype", None) == torch.float16) 
            or dtype == torch.float16
        ) and self.device in ["cpu", None]:
            logging.warning(
                "float16 models may not work on CPU. Consider using a GPU or bfloat16."
            )

    def override_config_with_kwargs(self, config: Any) -> Any:
        """
        Override configuration parameters with values from kwargs.
        Args:
            config: Configuration object to override
            config_overrides: Keyword arguments containing overrides
        Returns:
            config: The overridden configuration object
        >>> Example:
            >>> override_config_with_kwargs(config, {"n_layers": 12})
            HookedViTConfig(...)
        """
        for key, value in self.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def instantiate_model_class(self) -> Any:
        '''
        Instantiate the model class.
        Args:
            model_class: The class to instantiate
        Returns:
            An instance of the model class
        >>> Example:
            >>> instantiate_model_class(HookedViT)
            HookedViT(...)
        '''
        if self.model_class is None:
            if isinstance(self.model_type, ModelType):
                
                if self.model_type == ModelType.VISION:
                    from vit_prisma.models.base_vit import HookedViT

                    model_class = HookedViT
                else:  # TEXT
                    from vit_prisma.models.base_text_transformer import HookedTextTransformer

                    model_class = HookedTextTransformer
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")

        return model_class
        

    def initialise_model(self, model_class: Any, config: Any) -> Any:
        """
        Initialise the model with configuration and weights.
        Args:
            config: Configuration object for the model
            model_class: Class to instantiate the model
        Returns:
            Loaded model
        >>> Example:
            >>> initialise_model(config, HookedViT)
            HookedViT(...)
        """

        # Initialize model
        model = model_class(config)
        # Load weights if requested
        if self.pretrained:
            # Make sure local_path is included in kwargs for weight loading
            if self.local_path is not None:
                self.kwargs["local_path"] = self.local_path
            model.load_and_process_state_dict(
                self.state_dict,
                fold_ln=self.fold_ln,
                center_writing_weights=self.center_writing_weights,
                fold_value_biases=self.fold_value_biases,
                refactor_factored_attn_matrices=self.refactor_factored_attn_matrices,
            )

        model.to(device=self.device, dtype=self.dtype)
        model.set_use_attn_result(self.use_attn_result)

        if self.move_to_device:
            model.move_model_modules_to_device()

        logging.info(f"Loaded pretrained model {self.model_name} into HookedTransformer")
        return model





def create_config_object(model_name: str, model_type: ModelType) -> ConfigType:
    """
    Create a configuration object for a specific model.

    Args:
        model_name: Model name
        model_type: Model type

    Returns:
        HookedViTConfig or HookedTextTransformerConfig object
    >>> Example:
        >>> create_config_object("open-clip:timm/vit_base_patch16_clip_224.openai", ModelType.VISION)
        HookedViTConfig(...)
    """
    # Get raw configuration dictionary
    try:
        config_dict = MODEL_CONFIGS[model_type][model_name]
    except KeyError:
        raise ValueError(
            f"No configuration found for {model_name} with type {model_type}"
        )

    # Create appropriate config object
    if model_type == ModelType.VISION:
        return HookedViTConfig(**config_dict)
    else:
        return HookedTextTransformerConfig(**config_dict)



def fill_missing_keys(model, state_dict):
    """
    Fill in any missing keys with default initialization.

    Args:
        model: Model object
        state_dict: State dict with converted weights

    Returns:
        Complete state dict
    >>> Example:
        >>> fill_missing_keys(model, state_dict)
        {'weight_name': tensor(...), ...}
    """
    # Get the default state dict
    default_state_dict = model.state_dict()

    # Get missing keys
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())

    if missing_keys:
        logging.info(
            f"Filling in {len(missing_keys)} missing keys with default initialization"
        )

    for key in missing_keys:
        if "hf_model" in key:
            # Skip HuggingFace internal keys
            continue

        if "W_" in key:
            logging.warning(f"Missing key for weight matrix: {key}")

        state_dict[key] = default_state_dict[key]

    return state_dict


# ===============================
# Weight Loading Functions
# ===============================

class WeightLoader(ABC):
    """
    Abstract class to load weights for different model categories.
    """
    @abstractmethod
    def _load_weights(self, model_name: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Load weights for a specific model.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for weight loading
        Returns:
            Dictionary of model weights
        """
        pass

class TIMMWeightLoader(WeightLoader):
    """
    Class to load weights from TIMM models.
    """
    def __init__(self,
                 model_name: str,               
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.kwargs = kwargs


    def _load_weights(self)-> Dict[str, torch.Tensor]:
        """
        Loads weights from a TIMM model.
        Args:
            timm_model_name: Name of the TIMM model
            **kwargs: Additional arguments for TIMM model creation
        Returns:
            Dictionary of model weights for the specified TIMM model
        >>> Example:
            >>> _load_timm_weights("vit_base_patch16_224")
            {'patch_embed.proj.weight': tensor(...), 'patch_embed.proj.bias': tensor(...), ...}

        """
        try:
            import timm
        except ImportError:
            raise ImportError(
                "TIMM is required but not installed. Install with 'pip install timm'"
            )

        timm_model = timm.create_model(self.model_name, pretrained=True)
        for timm_param in timm_model.parameters():
            timm_param.requires_grad = False
        timm_weights = timm_model.state_dict()
        return timm_weights

class EVA02WeightLoader(WeightLoader):
    """
    Class to load weights from EVA02 models.
    """
    def __init__(self,
                 model_name: str,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.kwargs = kwargs

    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights for EVA02 models from TIMM especially.
        Args:
            eva02_model_name: Name of the EVA02 model
            **kwargs: Additional arguments for TIMM model creation
        Returns:
            Dictionary of model weights for the specified EVA02 model
        >>> Example:
            >>> _load_eva02_weights("open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b")
            {'patch_embed.proj.weight': tensor(...), 'patch_embed.proj.bias': tensor(...), ...}
        """
        try:
            import timm
        except ImportError:
            raise ImportError(
                "TIMM is required but not installed. Install with 'pip install timm'"
            )

        eva02_model_name_clean = self.model_name.split("open-clip:")[1]
        eva02_name, eva02_weights = eva02_model_name_clean.split(".")
        eva02_name = eva02_name.split("/")[1]
        eva02_model = timm.create_model(eva02_name, pretrained=eva02_weights)
        for eva02_param in eva02_model.parameters():
            eva02_param.requires_grad = False

        eva02_weights = eva02_model.state_dict()
        return eva02_weights

# timm_weights = [_load_timm_weights(timm_model_name) for timm_model_name in PASSING_MODELS["TIMM"]]

class CLIPWeightLoader(WeightLoader):
    """
    Class to load weights from CLIP models.
    """
    def __init__(self,
                 model_name: str,
                 dtype: torch.dtype = torch.float32,
                 local_path: Optional[str] = None,               
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.kwargs = kwargs
    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights from a CLIP model.
        Args:
            clip_model_name: Name of the CLIP model
        dtype: Data type for model parameters
        **kwargs: Additional arguments for CLIP model creation
    Returns:
        Dictionary of model weights for the specified CLIP model
    >>> Example:
        >>> _load_clip_weights("openai/clip-vit-base-patch16", torch.float32)
        {'vision_model.conv1.weight': tensor(...), 'vision_model.conv1.bias': tensor(...), ...}
    """

        clip_model = CLIPModel.from_pretrained(self.model_name, torch_dtype=self.dtype, **self.kwargs)
        for clip_param in clip_model.parameters():
            clip_param.requires_grad = False
        clip_weights = clip_model.state_dict()
        return clip_weights

# clip_weights = [_load_clip_weights(clip_model_name, torch.float32) for clip_model_name in PASSING_MODELS["CLIP"]]

class OpenCLIPWeightLoader(WeightLoader):
    """
    Class to load weights from OpenCLIP models.
    """
    def __init__(self,
                 model_name: str,
                 local_path: Optional[str] = None,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.local_path = local_path
        self.kwargs = kwargs

    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights from an OpenCLIP model.
        Args:
            open_clip_model_name: Name of the OpenCLIP model
            local_path: Optional local path to the model weights
            **kwargs: Additional arguments for OpenCLIP model creation
        Returns:
            Dictionary of model weights for the specified OpenCLIP model
        >>> Example:
            >>> _load_open_clip_weights("open-clip:timm/vit_base_patch16_clip_224.openai")
            {'visual.conv1.weight': tensor(...), 'visual.conv1.bias': tensor(...), ...}
        """
        if self.local_path is not None:
            checkpoint_path = self.local_path
        else:
            checkpoint_path = download_pretrained_from_hf(
                remove_open_clip_prefix(self.model_name),
                filename="open_clip_pytorch_model.bin",
            )
        open_clip_weights = load_state_dict(checkpoint_path)
        return open_clip_weights

# open_clip_weights = [_load_open_clip_weights(open_clip_model_name) for open_clip_model_name in PASSING_MODELS["OPEN_CLIP"]]


# eva02_weights = [_load_eva02_weights(eva02_model_name) for eva02_model_name in MODEL_CONFIGS["EVA02"].keys()]

class DINOWeightLoader(WeightLoader):
    """
    Class to load weights from DINO models.
    """
    def __init__(self,
                 model_name: str,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.kwargs = kwargs


    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights from a DINO model.
        Args:
            dino_model_name: Name of the DINO model
            dtype: Data type for model parameters
            **kwargs: Additional arguments for DINO model creation
        Returns:
            Dictionary of model weights for the specified DINO model
        >>> Example:
            >>> _load_dino_weights("facebook/dino-vits16", torch.float32)
            {'patch_embed.proj.weight': tensor(...), 'patch_embed.proj.bias': tensor(...), ...}
        """

        dino_model = ViTModel.from_pretrained(self.model_name, torch_dtype=self.dtype, **self.kwargs)
        for dino_param in dino_model.parameters():
            dino_param.requires_grad = False
        dino_weights = dino_model.state_dict()
        return dino_weights

# dino_weights = [_load_dino_weights(dino_model_name, torch.float32) for dino_model_name in PASSING_MODELS["DINO"]]

class ViViTWeightLoader(WeightLoader):
    """
    Class to load weights from ViViT models.
    """
    def __init__(self,
                 model_name: str,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.kwargs = kwargs

    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights from a ViViT model.
        Args:
            vivit_model_name: Name of the ViViT model
            dtype: Data type for model parameters
            **kwargs: Additional arguments for ViViT model creation
        Returns:
            Dictionary of model weights for the specified ViViT model
        >>> Example:
            >>> _load_vivit_weights("google/vivit", torch.float32)
            {'patch_embed.proj.weight': tensor(...), 'patch_embed.proj.bias': tensor(...), ...}
        """

        vivit_model = VivitForVideoClassification.from_pretrained(
            self.model_name, torch_dtype=self.dtype, **self.kwargs
        )
        for vivit_param in vivit_model.parameters():
            vivit_param.requires_grad = False
        vivit_weights = vivit_model.state_dict()
        return vivit_weights
 
class VJEPAWeightLoader(WeightLoader):
    """
    Class to load weights from VJEPA models.
    """
    def __init__(self,
                 model_name: str,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.kwargs = kwargs
    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Loads weights from a VJEPA model.
        Args:
            vjepa_model_name: Name of the VJEPA model
            **kwargs: Additional arguments for VJEPA model creation
        Returns:
            Dictionary of model weights for the specified VJEPA model.
        >>> Example:
            >>> _load_vjepa_weights("vit_prisma:vjepa_vit_base_patch16_224")
            {'patch_embed.proj.weight': tensor(...), 'patch_embed.proj.bias': tensor(...), ...}
        """
        try:
            from importlib import resources

            import yaml
            from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAModel
        except ImportError:
            raise ImportError(
                "VJEPA modules not found. Make sure vit_prisma.vjepa_hf is available."
            )

        with resources.open_text("vit_prisma.vjepa_hf", "paths_cw.yaml") as f:
            vivit_model_paths = yaml.safe_load(f)
        vjepa_model_path = vivit_model_paths[self.model_name]["loc"]
        vjepa_model = VJEPAModel.from_pretrained(vjepa_model_path)
        vjepa_weights = vjepa_model.state_dict()
        return vjepa_weights

class KandinskyWeightLoader(WeightLoader):
    """
    Class to load weights from Kandinsky models.
    """
    def __init__(self,
                 model_name: str,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.kwargs = kwargs
    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load weights from a Kandinsky model.
        Args:
            kandinsky_model_name: Name of the Kandinsky model
            **kwargs: Additional arguments for Kandinsky model creation
        Returns:
            Dictionary of model weights for the specified Kandinsky model
        >>> Example:
            >>> _load_kandinsky_weights("kandinsky-community/kandinsky-2-1-prior")
            {'image_encoder.conv1.weight': tensor(...), 'image_encoder.conv1.bias': tensor(...), ...}
        """

        kandinsky_model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_name,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            cache_dir="/network/scratch/s/sonia.joseph/diffusion",
        ).to("cuda")
        kandinsky_weights = kandinsky_model.state_dict()
        return kandinsky_weights

def create_weights_dict(
        timm_weights:Dict[str, torch.Tensor] = None,
        clip_weights:Dict[str, torch.Tensor] = None,
        open_clip_weights:Dict[str, torch.Tensor] = None,
        eva02_weights:Dict[str, torch.Tensor] = None,
        dino_weights:Dict[str, torch.Tensor] = None,
        vivit_weights:Dict[str, torch.Tensor] = None,
        vjepa_weights:Dict[str, torch.Tensor] = None,
        kandinsky_clip_weights:Dict[str, torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary of weight dictionaries for different model categories.
    Args:
        timm_weights (Dict[str, torch.Tensor], optional): Weights for TIMM models
        clip_weights (Dict[str, torch.Tensor], optional): Weights for CLIP models
        open_clip_weights (Dict[str, torch.Tensor], optional): Weights for OpenCLIP models
        eva02_weights (Dict[str, torch.Tensor], optional): Weights for EVA02 models
        dino_weights (Dict[str, torch.Tensor], optional): Weights for DINO models
        vivit_weights (Dict[str, torch.Tensor], optional): Weights for ViViT models
        vjepa_weights (Dict[str, torch.Tensor], optional): Weights for VJEPA models
        kandinsky_clip_weights (Dict[str, torch.Tensor], optional): Weights for Kandinsky CLIP models
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Dictionary containing weight dictionaries for each model category
    >>> For example:
        >>> create_weights_dict(timm_weights={'weight1': tensor(...), 'weight2': tensor(...)}, clip_weights={'weightA': tensor(...), 'weightB': tensor(...)})
        {'timm': {'weight1': tensor(...), 'weight2': tensor(...)}, 'clip': {'weightA': tensor(...), 'weightB': tensor(...)}, ...}
    """
    weights_dict = Dict(
        timm=timm_weights if 
                            timm_weights is not None 
                            else {},
        clip=clip_weights if 
                            clip_weights is not None 
                            else {},
        open_clip=open_clip_weights if 
                            open_clip_weights is not None 
                            else {},
        eva02=eva02_weights if 
                            eva02_weights is not None 
                            else {},
        dino=dino_weights if 
                            dino_weights is not None 
                            else {},
        vivit=vivit_weights if 
                            vivit_weights is not None 
                            else {},
        vjepa=vjepa_weights if 
                            vjepa_weights is not None 
                            else {},
        kandinsky_clip=kandinsky_clip_weights if 
                            kandinsky_clip_weights is not None 
                            else {},    
    )

# ===============================
# Helper Functions
# ===============================


def is_model_supported(model_name: str) -> bool:
    """Checks if a model is supported.
    Args:
        model_name (str): Name of the model to check

    Returns:
        bool: True if the model is supported, False otherwise
    >>> Example:
        >>> is_model_supported("openai/clip-vit-base-patch16")
        True
    """
    return model_name in MODEL_CATEGORIES


def get_supported_model_types(model_name: str) -> list:
    """Get the types supported by a model.
    Args:
        model_name (str): Name of the model

    Returns:
        list: List of supported model types
    >>> Example:
        >>> get_supported_model_types("openai/clip-vit-base-patch16")
        [<ModelType.VISION: 'VISION'>]
    """
    if not is_model_supported(model_name):
        return []

    types = [ModelType.VISION]  # All models support vision
    if model_name in TEXT_ONLY_SUPPORTED_MODELS:
        types.append(ModelType.TEXT)

    return types


#def list_available_models(
#    category: Optional[ModelCategory] = None,
#    model_type: Optional[ModelType] = None,
#    detailed: bool = False,
#) -> Union[List[str], Dict]:
#    """
#    List all available models, optionally filtered by category and model type.

#    Args:
#        category: Optional category filter
#        model_type: Optional model type filter
#        detailed: Whether to return detailed info (dict) or just names (list)

#    Returns:
#        List of model names or dictionary with detailed information
#    """
#    # Import the utilities
#    from vit_prisma.utils import list_available_models as list_models

#    format_type = "tabular" if detailed else "list"
#    return list_models(category, model_type, format=format_type)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a model.

    Args:
        model_name (str): Name of the model

    Returns:
        Dictionary with model details
    >>> Example:
        >>> get_model_info("openai/clip-vit-base-patch16")
        >>> {'model_name': 'openai/clip-vit-base-patch16', 'category': 'CLIP', 'type': 'VISION', 'config': {...}, 'weights': {...}}
    """
    from vit_prisma.utils import get_model_info as get_info

    return get_info(model_name)
