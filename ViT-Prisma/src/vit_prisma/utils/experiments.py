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

    def copy(self):
        '''
        Copies the experiment configuration.
        Returns:
            ExperimentConfig
        '''
        copy = self.__class__()
        for name, atr in vars(self).items():
            if type(atr) == list:
                setattr(copy, name, atr.copy()) # Set list attributes to a copy of the list.
            else:
                setattr(copy, name, atr) # Set other attributes to the same value.
        return copy
    
    def __str__(self):
        '''
        Returns a human-readable string representation of the experiment configuration.
        Returns:
            str
        '''
        str_print = f"--- {self.__class__.__name__} ---\n" # String print of the class.
        for name, atr in vars(self).items():
            attr = getattr(self, name) # Get attribute value.
            attr_str = f" * {name}:" # String representation of the attribute.
            
            if name == "mean_dataset" and self.compute_means and attr is not None:
                attr_str += get_sample_from_dataset(self.mean_dataset)
            elif name =="dataset" and attr is not None:
                attr_str += get_sample_from_dataset(self.dataset)
            else:
                atr_str += str(attr)
            atr_str += "\n"
            str_print += atr_str
        return str_print

    def __repr__(self) -> str:
        '''
        Returns an official string representation of the experiment configuration.
        '''
        return self.__str__()
    
def zero_fn(z: Tensor,hk):
    '''
    Returns zero tensor of the same shape as z.
    Args:
        z: Input tensor.
        hk: Hook context.
    Returns:
        Tensor
    '''
    return torch.zeros(z.shape)

def cst_fn(z: Tensor, cst: Tensor, hk):
    '''
    Returns a constant tensor of the same shape as z.
    Args:
        z: Input tensor.
        cst: Constant tensor.
        hk: Hook context.
    Returns:
        Tensor
    '''
    return cst[:z.shape[0], :z.shape[1], ...] # Match the shape of z.

def neg_fn(z: Tensor, hk):
    '''
    Returns the negative of the input tensor z.
    Args:
        z: Input tensor.
        hk: Hook context.
    Returns:
        Tensor
    '''
    return -z

class AblationConfig(ExperimentConfig):
    '''
    Class to handle ablation experiment configurations.
    '''
    def __init__(
            self,
            abl_type: str = "zero",
            mean_dataset: Any = None,
            compute_means: bool = False,
            batch_size: int = None,
            max_seq_len: int = None,
            abl_fn: Callable[[Tensor,Tensor,HookPoint], Tensor] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert abl_type in ["mean","zero","neg","random","custom"] # Assert valid ablation types.
        assert not (
            abl_type == "custom" and abl_fn is None
        ), "You must specify your ablation function"
        assert not (
            abl_type == "custom" and abl_fn is None
        ), "You must specify your ablation function" # Ensures custom ablation function is provided.
        assert not (abl_type == "random" and self.nb_metric_iteration < 0) # Ensures ablation type is not random when nb_metric_iteration is negative.
        assert not (abl_type != "random" and self.nb_metric_iteration != 1) # Ensures nb_metric_iteration is not 1 and ablation type is not random.
        assert not ( abl_type =="random" and not (cache_means)), "You must cache means for random ablation." # Ensures means are cached for random ablation.
        assert not (abl_type == "mean" and self.head_circuit in ["attn", "attn_scores"]), "Random ablation is not implemented for attention circuit" # Ensures random ablation is not used for attention circuit.

        if abl_type == "random" and (batch_size is None or max_seq_len is None):
            warnings.warn( "WARNINGS: Random and no shape specified for ablation. Will infer from the dataset. Use 'batch_size' and 'max_seq_len'to specify.")
        if abl_type == 'random' and self.nb_metric_iteration < 5:
            warnings.warn("WARNINGS: Random ablation with less than 5 iterations may lead to high variance in results.")
        
        self.abl_type = abl_type # Type of ablation.
        self.mean_dataset = mean_dataset # Dataset to compute means.
        self.compute_means = compute_means # Whether to compute means.
        self.batch_size = batch_size # Batch size for random ablation.
        self.max_seq_len = max_seq_len # Maximum sequence length for random ablation.

        if abl_type == "zero":
            self.abl_fn = zero_fn
        if abl_type == "neg":
            self.abl_fn = neg_fn
        if abl_type == "mean":
            self.abl_fn = cst_fn
        if abl_type == "custom" and abl_fn is not None:
            self.abl_fn = cst_fn # Can specify arbitrary ablation functions for custom ablation.

class PatchingConfig(ExperimentConfig):
    '''
    Configurations for patching experiments from the source dataset to the target dataset.
    '''
    def __init__(
            self, 
            source_dataset: List[str] = None,
            target_dataset: List[str] = None,
            patch_fn : Callable[[Tensor,Tensor,HookPoint], Tensor] = None,
            cache_act : bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_dataset = source_dataset # Source dataset for patching.
        self.target_dataset = target_dataset # Target dataset for patching.
        self.patch_fn = patch_fn # Patching function.
        self.cache_act = cache_act # Whether to cache activations.
        if patch_fn is None:
            self.patch_fn = cst_fn # Default patching function is constant function.

class Experiment:
    '''
    A virtual class to interactively apply hooks to layers or heads. The children class only needs to define the method get_hook.
    '''   
    def __init__(
            self,
            model: nn.Module,
            experiment_config: ExperimentConfig,
            experiment_metrics: ExperimentMetric,
    ):
        self.model = model
        self.experiment_config = experiment_config.adapt_to_model(model)
        self.experiment_metrics = experiment_metrics
        self.experiment_config.dataset = self.experiment_metrics.dataset
    
    def run_experiment(self):
        '''
        Runs the experiment.
        Returns:
            Tensor
        '''
        self.experiment_metrics.set_baseline(self.model)
        results = torch.empty(self.get_result_shape()) # Create an empty tensor to store results.
        if self.experiment_config.verbose:
            print(self.experiment_config)
        
        for layer in tqdm(range(self.experiment_config.beg_layer, self.experiment_config.end_layer)):
            if self.experiment_config.target_module == "attn_head":
                for head in self.experiment_config.heads:
                    hook = self.get_hook(
                        layer,
                        head,
                    )
                    results[layer, head] = self.compute_metric(hook).cpu().detach() # Compute metric and store in result tensor.
            else:
                hook = self.get_hook(
                    layer,
                )
                results[layer] = self.compute_metric(hook).cpu().detach() # Compute metric and store in result tensor.
            self.model.reset_hooks() # Reset hooks after each layer.
            if len(result.shape) < 2:
                results = result.unsqueeze(0) # Unsqueeze the first dimension if result shape is less than 2 to ensure we can easily plot the results
        return results

    def get_result_shape(self):
        '''
        Returns the shape of the result tensor.
        Returns:
            Tuple[int]
        '''
        if self.experiment_config.target_module == "attn_head":
            return (
                self.experiment_config.end_layer - self.experiment_config.beg_layer,
                len(self.experiment_config.heads),
            ) + self.experiment_metrics.shape
        else:
            return (
                self.experiment_config.end_layer - self.experiment_config.beg_layer,
            ) + self.experiment_metrics.shape
    def compute_metric(self, abl_hook):
        '''
        Computes the metric using the provided hook.
        Args:
            ablhook: Ablation hook to apply during metric computation.
        Returns:
            Tensor
        '''
        mean_metric = torch.zeros(self.experiment_metrics.shape)
        self.model.reset_hooks()
        hk_name, hk = abl_hook
        self.model.add_hook(hk_name, hk)

        # Only useful if the computation are stochastic. In most cases, only 1 loop. 
        for it in range(self.experiment_config.nb_metric_iteration):
            self.update_setup(hk_name)
            mean_metric += self.experiment_metrics.compute_metric(self.model)
        return mean_metric / self.experiment_config.nb_metric_iteration

    def update_setup(self, hk_name):
        '''
        Updates the setup of the hook.
        Args:
            hk_name: Name of the hook to update.
        Returns:
            None
        '''
        pass

    def get_target(self, layer: int, head: int = None, target_module: str = None)-> Tuple[str,int]:
        '''
        Returns the target module name for the given layer and head (pass target_module to override config settings).
        Args:
            layer: Layer index.
            head: Head index.
            target_module: Target module name.
        Returns:
            str
        '''
        if head is not None:
            hook_name = f"blocks.{layer}.attn.hook_{self.experiment_config.head_circuit}"
            dim = (
                1 if "hook_attn" in hook_name else 2
            ) # Equate dimension to 1 for attn and 2 for others.
        else:
            if self.experiment_config.target_module == "mlp" or target_module == "mlp":
                hook_name = f"blocks.{layer}.mlp.hook_out"
            else:
                hook_name = f"blocks.{layer}.hook_attn_out"
        return hook_name, dim
    
class AblationExperiment(Experiment):
    '''
    Runs an ablation experiment according to the ablation configuration.
    Passes semantic_indices not None to average across different index positions. 
    
    '''
    def __init__(
            self,
            model:nn.Module,
            config: AblationConfig,
            metric: ExperimentMetric,
            semantic_indices: List[int] = None,
            means_by_groups : bool = False,
            groups: List[List[int]] = None,
    ):
        super().__init__(model, config, metric)
        assert "AblationConfig" is str(type(config)), "Config must be of type AblationConfig"
        assert not (
            semantic_indices is not None
        ) and (config.head_circuit in ["hook_attn_scores", "hook_attn"]) # not implemented for attention scores or attn ablation.
        assert not (means_by_groups and groups is None)
        self.semantic_indices = semantic_indices
        self.means_by_groups = means_by_groups
        self.groups = groups # list of (list of indices of element of the group)

        if self.semantic_indices is not None:
            warnings.warn("semantic_indices is not None. It is probably what you want.")
            self.max_len = max([len(self.model.tokenizer(t).input_ids) for t in self.experiment_config.dataset])
            self.get_seq_no_sem(self.max_len)

            if self.experiment_config.mean_dataset is None and experiment_config.compute_means:
                self.experiment_config.mean_dataset = self.experiment_config.dataset
            
            if self.experiment_config.abl_type == "random":
                if self.experiment_config.batch_size is None:
                    self.experiment_config.batch_size = len(self.experiment_config.dataset)
                if self.experiment_config.max_seq_len is None:
                    self.experiment_config.batch_size = max(
                        [len(self.model.tokenizer(t).input_ids) for t in self.experiment_config.dataset]
                    ) # Infer max_seq_len from dataset
                if self.experiment_config.cache_means and self.experiment_config.compute_means:
                    self.get_all_means()

    def run_ablation(self):
        '''
        Runs the ablation experiment.
        Returns:
            Tensor
        '''
        return self.run_experiment()
    
    def get_hook(self, layer: int, head: int = None, target_module: str = None):
        '''
        Returns the ablation hook for the given layer and head.
        Args:
            layer: Layer index.
            head: Head index.
            target_module: Target module name.
        Returns:
            Tuple[str, Callable[[Tensor, HookPoint], Tensor]]
        '''
        hk_name, dim = self.get_target(layer, head, target_module)
        mean = None
        if self.config.compute_means:
            if self.experiment_config.compute_means:
                if self.means_by_groups:
                    mean = self.mean_cache[hk_name]
                else:
                    mean = self.get_mean(hk_name)
        
        abl_hook = get_act_hook(
            self.experiment_config.abl_fn, mean, head, dim=dim
        ) # Get activation hook as ablation hook
        return (hk_name,abl_hook)
    
    def get_all_mean(self):
        '''
        Gets all the means in cache. 
        Returns:
            None
        '''
        self.act_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.act_cache)
        self.model(self.config.mean_dataset)
        self.mean_cache = {}
        for hk in self.act_cache.keys():
            if "blocks" in hk:
                self.mean_cache[hk] = self.compute_mean(self.act_cache[hk], hk)
    
    def get_mean(self,hook_name):
        '''
        Gets mean.
        Args:
            hook_name: A name of hook to get mean from.
        Returns:
            None
        '''
        cache = {}
    
        def cache_hook(z: Tensor,hook:Tensor):
            '''
            Caches hook.
            Args:
                z(Tensor): An input tensor.
                hook (Tensor): A tensor of a hook point.
            Returns:
                None
            '''
            cache[hook_name] = z.detach().to("cuda") # Cache hook 
        
        self.model.reset_hooks() # Reset hooks 
        self.model.run_with_hooks(
            self.config.mean_dataset. fwd_hooks=[(hook_name, cache_hook)]
        )
        return self.compute_mean(cache[hook_name], hook_name)
    
    def compute_mean(self,
                     z: Tensor,
                     hk_name: str):
        '''
        Computes mean from input tensors.
        Args:
            z(Tensor): An input tensor.
            hk_name(str): A string of hook name.
        Returns:
            a float or tensor of mean.
        '''
        mean = (
            torch.mean(z. dim=0, keepdim=False).detach().clone()
        ) # Compute mean along the batch dim
        mean = einops.repeat(mean, "... -> s ...", s=z.shape[0])

        if self.config.abl_type == "random":

            mean = get_ramdom_sample(
                z.clone().flatten(start_dim=0, end_dim=1), # Clone (return a copy with flowing gradients) and flatten (compress multidimensional arrays into one-dimensional ones)
            
            (
                self.config.batch_size,
                self.config.max_seq_len,
            ),
            ) 

        if self.means_by_groups:
            mean = torch.zeros_like(z)
            for group in self.groups:
                group_mean = torch.mean(z[group], dim=0, keepdim=False).detach().clone()
                mean[group] = einops.repeat(group_mean,"... -> s ...",s=len(group))

        if (
            self.semantic_indices is None
            or "hook_attn" in hk_name
            or self.means_by_groups
        ):
            return mean
        
        dataset_length = len(self.config.mean_dataset)

        for semantic_symbol, semantic_indices in self.semantic_indices.items():
            mean[list(range(dataset_length)), semantic_indices] = einops.repeat(
                torch.mean(
                    z[list(range(dataset_length)), semantic_indices],
                    dim=0,
                    keepdim=False,
                ).clone(),
                "... -> s ...",
                s=dataset_length, 
                ) # Compute the global mean of the semantic-position and write the same mean into every sequence's semantic position

    def get_seq_no_sem(self, max_len: int):
        '''
        Obtains the sequence without semantics.
        Args:
            max_len(int): An integer of maximum length.
        Returns: 
            None
        '''
        self.seq_no_sem = []
        for pos in range(max_len):
            seq_no_sem_at_pos = []
            
            for seq in range(len(self.config.mean_dataset)):
                seq_is_sem = False
                for semantic_symbol, semantic_indices in self.semantic_indices.items():
                    if pos == semantic_indices[seq]:
                        seq_is_sem = True # If the position is the same as the semantic index of sequence, sequence is semantic
                        break
                
                if self.semantic_indices["end"][seq] < pos:
                    seq_is_sem = True 






def get_act_hook(
        fn,
        alt_act: Any=None,
        idx: Any = None,
        dim: Any = None,raise ValueError("You must specify batch_size and max_seq_len for random ablatio
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

