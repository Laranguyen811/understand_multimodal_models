# %%
from typing import Callable, Union, List, Tuple, Any
import torch
import warnings
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import random
import time
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from functools import *
import pandas as pd
import gc 
import collections
import copy
import itertools
from transformers import AutoModelForObjectDetection, AutoConfig, AutoTokenizer
from vit_prisma.prisma_tools.hook_point import HookPoint
from vit_prisma.utils.experiment_utils import calculate_gelu, to_numpy, get_corner, print_gpu_mem, get_sample_from_dataset 

# %%
class ExperimentMetric:
    def __init__(
            self, 
            metric: Callable[[Any], torch.Tensor],
            dataset: Any,
            scalar_metric: bool = True,
            relative_metric:bool=True,

    ):
        '''
        Class to handle experiment metrics.
        '''
        self.relative_metric = relative_metric
        self.metric = metric # Metric can return any tensor shape. Can call run_with_hook with reset_hooks_start=False.
        self.scalar_metric = scalar_metric # Whether the metric returns a scalar value.
        self.baseline = None # Baseline value for relative metrics.
        self.daytaet = dataset # Dataset to run the metric on.
        self.shape = None # Shape of the metric output.

        def set_baseline(self, model: nn.Module):
            '''
            Sets the baseline value for relative metrics.
            Args:
                model: Model to run the metric on.
            Returns:
                None
            '''
            model.reset_hooks()
            base_metric = self.metric(model,self.dataset)
            self.shape = base_metric.shape
        
        def compute_metric(self,model):
            assert (self.baseline is not None) or not (
                self.relative_metric
            ), "Baseline must be set for relative mean."
            out = self.metric(model,self.dataset)
            if self.scalar_metric:
                assert (
                    len(out.shape) == 0
                ), "Output of scalar metric has shape of length > 0."

            self.shape = out.shape
            if self.relative_metric:
                out = out / self.baseline - 1
            return out

class ExperimentConfig:
    def __init__(
            self, 
            target_module: str = "attn_head",
            layers: Union[Tuple[int,int], str] = "all",
            heads: Union[List[int], str] = "all",
            verbose: bool = False,
            head_circuit: str = "z",
            nb_metric_iteration: int = 1,
    ):
        '''
        Class to handle experiment configurations.
        '''
        assert target_module in ["mlp","attn_layer","attn_head"], f"Invalid target module: {target_module}"
        assert head_circuit in ["q","k","v","z","attn","attn_scores","result"], f"Invalid head circuit: {head_circuit}"
        self.target_module = target_module # Target module to run the experiment on.
        self.layers = layers # Layers to run the experiment on.
        self.heads = heads # Heads to run the experiment on.
        self.verbose = verbose # Whether to print verbose output.
        self.head_circuit = head_circuit # Head circuit to use.
        self.dataset = dataset # Dataset to run the experiment on.

        self.beg_layer = None # Beginning layer for the experiment.
        self.end_layer = None # End layer for the experiment.

    def adapt_to_model(
            self,
            model: nn.Module,
    ):
        '''
        Adapts the experiment configuration to the model.
        Args:
            model: Model to adapt the configuration to.
        Returns:
            None
        '''
        model_cfg = self.copy()
        if self.target_module == "attn_head":
            if self.heads == "all":
                model_cfg.heads = list(range(model.cfg.num_heads))
        if self.layers == "all":
            model_cfg.beg_layer = 0
            model_cfg.end_layer = model.cfg.n_layers
        else:
            model_cfg.beg_layer, model_cfg.end_layer = self.layers
        return model_cfg

    
def get_act_hook(
        fn,
        alt_act: Any=None,
        idx: Any = None,
        dim: Any = None,
        name: Any = None,
        message: Any = None,
        metadata: Any = None, 

):
    '''
    Returns a hook that modifies the activation on the fly. 
    Args:
        fn: Function.
        alt_act: Alternative activations (tensor of the same shape of z).
        idx: Index of the head.
        dim: Dimension of the tensor.
        name: Name of the activation.
        message: Message.
        metadata: metadata of the activation. 
    Returns: 
        Tensor
    '''
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name
            hook.ctx["metadata"] = metadata

            if message is not None:
                print(message)
            
            if (
                dim is None
            ): # Mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z,alt_act,hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z
        
    else:

        def custom_hook(z, hook):
            '''
            Create custom hooks. 
            '''
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name
            hook.ctx["metadata"] = metadata
        
            if message is not None:
                print(message)
            
            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], hook)
            return z
        
    return custom_hook

