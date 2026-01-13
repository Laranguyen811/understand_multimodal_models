# %%
from typing import Callable, Union, List, Tuple, Any, Optional, Dict
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
from abc import ABC, abstractmethod
# %%
class ExperimentMetric:
    def __init__(
            self, 
            metric,
            dataset,
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
        self.dataset = dataset # Dataset to run the metric on.
        self.shape = None # Shape of the metric output.

    def set_baseline(self, model):
        '''
        Sets the baseline value for relative metrics.
        Args:
            model: Model to run the metric on.
        Returns:
            None
        '''
        model.reset_hooks()
        base_metric = self.metric(model,self.dataset)
        self.baseline = base_metric
        self.shape = base_metric.shape
        
    def compute_metric(self,model):
        assert (self.baseline is not None) or not (
            self.relative_metric
        ), "Baseline must be set for relative mean."
        out = self.metric(model, self.dataset)
        if self.scalar_metric:
            assert (
                len(out.shape) == 0
            ), "Output of scalar metric has shape of length > 0."

        self.shape = out.shape
        if self.relative_metric:
            out = (out / self.baseline) - 1
        return out

class ExperimentConfig:
    def __init__(
            self, 
            abl_type:str="zero",
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
        self.nb_metric_iteration: int = nb_metric_iteration
        self.target_module: str = target_module # Target module to run the experiment on.
        self.layers: Union[Tuple[int,int], str] = layers # Layers to run the experiment on.
        self.heads: Union[List[int], str] = heads # Heads to run the experiment on.
        self.verbose: bool = verbose # Whether to print verbose output.
        self.head_circuit: str = head_circuit # Head circuit to use.
        self.dataset: Optional[Any] = None # Set dataset to None.
        self.mean_dataset: Optional[Any] = None # Set mean dataset to None.
        self.batch_size: Optional[int] = None # Set batch size to None.
        self.beg_layer: Optional[int] = None # Beginning layer for the experiment.
        self.end_layer: Optional[int] = None # End layer for the experiment.
        self.abl_type: str = abl_type
        
        self.compute_means: bool = (abl_type == "mean" or abl_type == "custom" or abl_type == "random")
    def adapt_to_model(
            self,
            model,
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
            elif isinstance(model_cfg.heads, str):
                model_cfg.heads = list(range(model.cfg.num_heads))
        if self.layers == "all":
            model_cfg.beg_layer = 0
            model_cfg.end_layer = model.cfg.n_layers
        else:
            if isinstance(self.layers, tuple):
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
            attr_str = f"* {name}:" # String representation of the attribute.
            
            if name == "mean_dataset" and self.compute_means and attr is not None:
                attr_str += get_sample_from_dataset(self.mean_dataset)
            elif name =="dataset" and attr is not None:
                attr_str += get_sample_from_dataset(self.dataset)
            else:
                attr_str += str(attr)
            attr_str += "\n"
            str_print += attr_str
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
            cache_means: bool = False,
            batch_size: Optional[int] = None,
            max_seq_len: Optional[int] = None,
            abl_fn: Optional[Callable[[Tensor,Tensor,HookPoint], Tensor]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert abl_type in ["mean","zero","neg","random","custom"] # Assert valid ablation types.
        assert not (
            abl_type == "custom" and abl_fn is None
        ), "You must specify your ablation function" # Ensures custom ablation function is provided.
        assert not (abl_type == "random" and self.nb_metric_iteration < 0) # Ensures ablation type is not random when nb_metric_iteration is negative.
        assert not (abl_type != "random" and self.nb_metric_iteration == 1) # Ensures nb_metric_iteration is not 1 and ablation type is not random.
        assert not ( abl_type =="random" and not (cache_means)), "You must cache means for random ablation." # Ensures means are cached for random ablation.
        assert not (abl_type == "mean" and self.head_circuit in ["attn", "attn_scores"]), "Random ablation is not implemented for attention circuit" # Ensures random ablation is not used for attention circuit.

        if abl_type == "random" and (batch_size is None or max_seq_len is None):
            warnings.warn( "WARNINGS: Random and no shape specified for ablation. Will infer from the dataset. Use 'batch_size' and 'max_seq_len'to specify.")
        if abl_type == 'random' and self.nb_metric_iteration < 5:
            warnings.warn("WARNINGS: Random ablation with less than 5 iterations may lead to high variance in results.")
        
        self.abl_type = abl_type # Type of ablation.
        self.mean_dataset = mean_dataset # Dataset to compute means.
        self.cache_means = cache_means # Whether to cache means.
        self.compute_means = (abl_type == "mean" or abl_type == "custom" or abl_type == "random")
        
        self.abl_fn = abl_fn

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
            source_dataset: Optional[List[str]] = None,
            target_dataset: Optional[List[str]] = None,
            patch_fn : Optional[Callable[[Tensor,Tensor,HookPoint], Tensor]] = None,
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

class Experiment(ABC):
    '''
    A virtual class to interactively apply hooks to layers or heads. The children class only needs to define the method get_hook.
    '''   
    def __init__(
            self,
            model,
            experiment_config: ExperimentConfig,
            experiment_metrics: ExperimentMetric,
    ):
        self.model = model
        self.cfg = experiment_config.adapt_to_model(model)
        self.experiment_metrics = experiment_metrics
        self.cfg.dataset = self.experiment_metrics.dataset
    
    @abstractmethod
    def get_hook(self, layer: int, head: Optional[Union[int, str]] = None, target_module: Optional[str]=None) -> Tuple[str, Callable]:
        '''
        Returns the hook for the given layer and head.
        Must be implemented by child classes.
        '''
        pass
    def run_experiment(self):
        '''
        Runs the experiment.
        Returns:
            Tensor
        '''
        self.experiment_metrics.set_baseline(self.model)
        result_shape = self.get_result_shape()
        assert result_shape is not None, "Result shape cannot be None"
        results = torch.empty(result_shape) # Create an empty tensor to store results.
        if self.cfg.verbose:
            print(self.cfg)
        
        if self.cfg.beg_layer is not None and self.cfg.end_layer is not None:
            for layer in tqdm(range(self.cfg.beg_layer, self.cfg.end_layer)):
                if self.cfg.target_module == "attn_head":
                    for head_idx, head in enumerate(self.cfg.heads):
                        if isinstance(head, int) and head is not None:
                            hook = self.get_hook(
                                layer,
                                head,
                            )
                            metric_result = self.compute_metric(hook)
                            if metric_result is not None:
                                results[layer, head_idx] = metric_result.cpu().detach() # Compute metric and store in result tensor.
                else:
                    hook = self.get_hook(
                        layer,
                    )
                    metric_result = self.compute_metric(hook)
                    if metric_result is not None:
                        results[layer] = metric_result.cpu().detach() # Compute metric and store in result tensor.
                self.model.reset_hooks() # Reset hooks after each layer.
                if len(results.shape) < 2:
                    results = results.unsqueeze(0) # Unsqueeze the first dimension if result shape is less than 2 to ensure we can easily plot the results
        return results

    def get_result_shape(self):
        '''
        Returns the shape of the result tensor.
        Returns:
            Tuple[int]
        '''
        metric_shape = self.experiment_metrics.shape if self.experiment_metrics.shape is not None else ()
        if self.cfg.beg_layer is not None and self.cfg.end_layer is not None:
            if self.cfg.target_module == "attn_head":
            
                return (
                    self.cfg.end_layer - self.cfg.beg_layer,
                    len(self.cfg.heads),
                ) + metric_shape
            else:
                return (
                    self.cfg.end_layer - self.cfg.beg_layer,
                ) + metric_shape
        
    def compute_metric(self, abl_hook):
        '''
        Computes the metric using the provided hook.
        Args:
            ablhook: Ablation hook to apply during metric computation.
        Returns:
            Tensor
        '''
        mean_metric = torch.zeros(self.experiment_metrics.shape) if self.experiment_metrics.shape is not None else None
        self.model.reset_hooks()
        hk_name, hk = abl_hook
        self.model.add_hook(hk_name, hk)

        # Only useful if the computation are stochastic. In most cases, only 1 loop. 
        for it in range(self.cfg.nb_metric_iteration):
            self.update_setup(hk_name)
            metric = self.experiment_metrics.compute_metric(self.model)
            if mean_metric is None:
                mean_metric = metric
            else:
                mean_metric += metric
        
        return mean_metric / self.cfg.nb_metric_iteration if mean_metric is not None else mean_metric


    def update_setup(self, hk_name):
        '''
        Updates the setup of the hook.
        Args:
            hk_name: Name of the hook to update.
        Returns:
            None
        '''
        pass

    def get_target(self, layer: int, head: Optional[Union[int, str]]= None, target_module: Optional[str] = None):
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
            hook_name = f"blocks.{layer}.attn.hook_{self.cfg.head_circuit}"
            dim = (
                1 if "hook_attn" in hook_name else 2
            ) # Equate dimension to 1 for attn and 2 for others.
        else:
            if self.cfg.target_module == "mlp" or target_module == "mlp":
                hook_name = f"blocks.{layer}.mlp.hook_out"
            else:
                hook_name = f"blocks.{layer}.hook_attn_out"
        dim = None # All the activation dimensions are ablated
        return hook_name, dim
    
class Ablation(Experiment):
    '''
    Runs an ablation experiment according to the ablation configuration.
    Passes semantic_indices not None to average across different index positions. 
    
    '''
    def __init__(
            self,
            model: nn.Module,
            ablation_config: AblationConfig,
            metric: ExperimentMetric,
            semantic_indices: Optional[Dict[str, List[int]]] = None,
            means_by_groups : bool = False,
            groups: Optional[List[List[int]]] = None,
    ):
        super().__init__(model, ablation_config, metric)
        assert "AblationConfig" in str(type(ablation_config)) or "PatchingConfig" in str(type(ablation_config)), "Config must be of type AblationConfig or PatchingConfig"
        
        if semantic_indices is not None:
            assert ablation_config.head_circuit not in ["hook_attn_scores", "hook_attn"],  # not implemented for attention scores or attn ablation.
            "Semantic indices ablation not implemented for attention scores or attn."
        assert not (means_by_groups and groups is None)
        self.semantic_indices = semantic_indices
        self.means_by_groups = means_by_groups
        self.groups = groups # list of (list of indices of element of the group)
        self.ablation_config = ablation_config
        if self.semantic_indices is not None:
            warnings.warn("semantic_indices is not None. It is probably what you want.")
            
            if self.cfg.mean_dataset is None and ablation_config.compute_means:
                self.cfg.mean_dataset = self.experiment_metrics.dataset
            
            if self.cfg.mean_dataset is not None:
                self.max_len = max([len(self.model.tokenizer(t).input_ids) for t in self.cfg.mean_dataset])
                self.get_seq_no_sem(self.max_len)
            
            if self.cfg.abl_type == "random":
                if self.cfg.batch_size is None:
                    self.cfg.batch_size = len(self.experiment_metrics.dataset)
                if self.ablation_config.max_seq_len is None:
                    self.cfg.batch_size = max(
                        [len(self.experiment_metrics.dataset[i]) for i in range(len(self.experiment_metrics.dataset))]
                    ) # Infer max_seq_len from dataset
                if ablation_config.cache_means and self.cfg.compute_means:
                    self.get_all_mean()

    def run_ablation(self):
        '''
        Runs the ablation experiment.
        Returns:
            Tensor
        '''
        return self.run_experiment()
    
    def get_hook(self, layer: int, head: Optional[Union[int, str]] = None, target_module: Optional[str] = None):
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
        if self.cfg.compute_means:
            if isinstance(self.ablation_config, AblationConfig) and self.ablation_config.cache_means:
                if self.means_by_groups:
                    mean = self.mean_cache[hk_name]
                else:
                    mean = self.get_mean(hk_name)
        
        abl_hook = get_act_hook(
            self.ablation_config.abl_fn, mean, head, dim=dim
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
        self.model(self.cfg.mean_dataset)
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
            self.cfg.mean_dataset, fwd_hooks=[(hook_name, cache_hook)]
        )
        return self.compute_mean(cache[hook_name], hook_name)
    
    def compute_mean(self,
                     z: Tensor,
                     hk_name: str) -> Tensor:
        '''
        Computes mean from input tensors.
        Args:
            z(Tensor): An input tensor.
            hk_name(str): A string of hook name.
        Returns:
            a float or tensor of mean.
        '''
        global_mean = z.mean(dim=0, keepdim=False).detach() # Compute mean along the batch dim
        mean = global_mean.unsqueeze(0).expand(z.shape[0], -1, -1).clone() # Expand back to batch dimension. Shape [seq, d_model] -> [batch, seq, d_model]

        if self.cfg.abl_type == "random":
            mean = get_random_sample(
                z.view(-1, z.shape[-1]),
            
            (
                self.cfg.batch_size,
                self.ablation_config.max_seq_len,
            ),
            ) 

        if self.means_by_groups and self.groups is not None:
            mean = torch.zeros_like(z)
            for group in self.groups:
                group_mean = torch.mean(z[group], dim=0, keepdim=False).detach().clone() # Create group mean
                mean[group] = einops.repeat(group_mean,"... -> s ...",s=len(group)) # Create the mean of the specified group

        if (
            self.semantic_indices is not None
            and "hook_attn" not in hk_name
            and not self.means_by_groups
        ):
            batch_idx = torch.arange(z.shape[0], device=z.device)
            for symbol, indices in self.semantic_indices.items():
                # Vectorised mean of specific positions across the batch
                sem_mean = z[batch_idx, indices].mean(dim=0, keepdim=True)
        return mean
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
            if self.cfg.mean_dataset is not None and self.semantic_indices is not None:
                for seq in range(len(self.cfg.mean_dataset)):
                    seq_is_sem = False
                    for semantic_symbol, semantic_indices in self.semantic_indices.items():
                        if pos == semantic_indices[seq]:
                            seq_is_sem = True # If the position is the same as the semantic index of sequence, sequence is semantic
                            break
                
                    if self.semantic_indices["end"][seq] < pos:
                        seq_is_sem = True # If semantic indices at end sequence is smaller than the current position, sequence is semantic
                    
                    if not (seq_is_sem):
                        seq_no_sem_at_pos.append(seq) # If sequence is not semantic, append sequence to the seq_no_sem_at_pos list

            self.seq_no_sem.append(seq_no_sem_at_pos.copy()) # Append the copy of seq_no_sem_at_pos to seq_no_sem

    def update_setup(self, hk_name):
        '''
        Updates the setup.
        Args:
            hk_name(str): A string of hook name
        Returns:
            None
        ''' 
        if self.cfg.abl_type == "random":
                self.mean_cache[hk_name] = self.compute_mean(
                    self.act_cache[hk_name], hk_name
                ) # Randomise the cache for random ablation

class Patching(Experiment):
    def __init__(
            self, 
            model: nn.Module,
            patching_config: PatchingConfig,
            metric: ExperimentMetric
    ):
        super().__init__(model, patching_config, metric)
        assert "PatchingConfig" in str(type(patching_config))
        self.patching_config = patching_config
        if self.patching_config.cache_act:
            self.get_all_act()
        
    def run_patching(self):
        '''
        Runs patching.
        '''
        return self.run_experiment()
    
        
    def get_hook(self, layer: int, head:Optional[Union[int, str]]=None, target_module:Optional[str]=None, patch_fn:Optional[Callable]=None)-> Tuple:
        '''
        Gets hook.
        Args:
            layer (int): An integer of a layer number.
            head(Optional[str]): A string of a head name.
            target_module (Optional[str]): A str of a target module name.
            patch_fn(Optional[Callable]): A patching function.
        Returns:
            Tuple of hook name and hook.
        '''
        hook_name, dim = self.get_target(layer, head, target_module=target_module) # Get the hook name and the dimension
        if self.patching_config.cache_act:
            act = self.act_cache[hook_name] # Get activation on the source dataset if config has cache_act
        else:
            act = self.get_act(hook_name)  # Otherwise, get activation from hook name
            
        if patch_fn is None:
            patch_fn = self.patching_config.patch_fn # if patch function is None, use patch_fn in config
            
        hook = get_act_hook(self.patching_config.patch_fn, act, head, dim=dim) # Get the activation hook
        return(hook_name, hook)
    def get_all_act(self):
            '''
            Gets all activations. 
            '''
            self.act_cache = {}
            self.model.reset_hooks() # Reset hooks
            self.model.cache_all(self.act_cache) # Cache all activations
            self.model(self.patching_config.source_dataset) # Use the model with source dataset

    def get_act(self, hook_name: str):
        '''
        Gets activations.
        '''    
        cache = {}

        def cache_hook(z: Tensor, hook: Tensor):
            '''
            Caches hooks.
            Args:
                z(Tensor): A tensor of inputs.
                hook(Tensor): A tensor of hook.
            Returns:
                ActivationCache of hook.
            '''
            cache[hook_name] = z.detach().to("cuda")

            self.model.reset_hooks()
            self.model.run_with_hooks(
                self.patching_config.source_dataset, fwd_hooks=[(hook_name, cache_hook)]
                )
        return cache[hook_name]


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

def get_random_sample(
        activation_set,
        output_shape
):
    '''
    Generates a tensor of shape (batch, seq_len, ...) made of vectors sampled from activation_set.
    Args:
        activation_set: An activation set of shape (N, ...).
        output_shape: An output shape.
    Returns:
        A tensor of output
    '''
    N = activation_set.shape[0]
    orig_shape = activation_set.shape[1:] # Assigns the original shape to activation set shape from index 1 onwards
    batch, seq_len = output_shape 
    idx = torch.randint(low=0, high=N, size=(batch * seq_len,)) # Create an index from a random integer with specified parameters
    out = activation_set[idx].clone() # Clone the activation set at the specific index as output
    out = out.reshape((batch, seq_len) + orig_shape) # Reshape output
    return out

