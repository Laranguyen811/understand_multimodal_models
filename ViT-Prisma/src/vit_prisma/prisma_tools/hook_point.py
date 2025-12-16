"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Sequence, Literal
from vit_prisma.prisma_tools.lens_handle import LensHandle
import torch
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
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from functools import *
import pandas as pd
import gc
import collections
import copy
import itertools
from vit_prisma.prisma_tools.activation_cache import ActivationCache 
import logging


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (adapted from TransformerLens, which was inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """
    def __init__(self):
        super().__init__()
        self.fwd_hooks: List[LensHandle] = []
        self.bwd_hooks: List[LensHandle] = []

        self.ctx = {} # Acts as a communication channel between the forward and backward passes of the custom autogra fnction

        self.name: str
    
    def add_perma_hook(self, hook, dir="fwd") -> None:
        self.add_hook(hook, dir, is_permanent=True)
    
    def add_hook(
            self, hook, dir="fwd", is_permanent=False, level=None, prepend=False
    ) -> None:
        """
        If prepend is True, add this hook before all other hooks.
        """

        if dir == "fwd":

            def full_hook(module, module_input, module_output):
                return hook(module_output, hook=self)

            full_hook.__name__ = (
                hook.__repr__()
            )

            handle = self.register_forward_hook(full_hook)
            handle = LensHandle(handle, is_permanent, level)

            if prepend:
                self._forward_hooks.move_to_end(handle.hook.id, last=False)
                self.fwd_hooks.insert(0, handle)
            else:
                self.fwd_hooks.append(handle)

        elif dir == "bwd":

            def full_hook(module, module_input, module_output):
                return hook(module_output[0], hook=self)
            
            full_hook.__name__ = (
                hook.__repr__()
            )

            handle = self.register_backward_hook(full_hook)
            handle = LensHandle(handle, is_permanent, level)
        
            if prepend:
                self._backward_hooks.move_to_end(handle.hook.id, last=False)
                self.bwd_hooks.insert(0, handle)
            else:
                self.bwd_hooks.append(handle)
        
        else :
            raise ValueError(f"Invalid dir {dir}. dir must be 'fwd' or 'bwd'")

    def remove_hooks(self, dir="fwd", including_permanent=False, level=None) -> None:
        def _remove_hooks(handles: List[LensHandle]) -> List[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles
        
        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}. dir must be 'fwd', 'bwd', or 'both'")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x):
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        split_name = self.name.split(".")
        return int(split_name[1])

# Define type aliases
NamesFilter = Optional[Union[Callable[[str],bool], Sequence[str]]]


class HookedRootModule(nn.Module):
    '''
    A class building on nn.Module to interface nicely with hook points.
    Adds various nice utilities, most notably run_with_hooks to run the model with temporary hooks, and run_with_cache to run the model on some input and return a cache of all activations. 
    
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_caching = False
    
    def set_up(self):
        '''
        Adds a parameter to each module given its name and builds a dictionary mapping a module name to the module.
        Setup function needs to run in __init__ after defining all.
        
        '''
        self.mod_dict = {}
        self.hook_dict = {}

        for name, module in self.named_modules(): # Loop through name and module in named modules
            module.name = name
            self.mod_dict[name] = module # Assign value of the name key to module 
            if "HookPoint" in str(type(module)): 
                self.hook_dict[name] = module # If HookPoint is a module, assign the value of name to module
            
    def hook_points(self):
        return self.hook_dict.values() # Return the values of hook dictionary
    
    def remove_all_hook_fns(self, direction="both"):
        '''
        Removes all hook functions.
        '''
        for hp in self.hook_points():
            hp.remove_hooks() # Remove all hook points
    
    def clear_contexts(self):
        '''
        Clear contexts in hook points.
        
        '''
        for hp in self.hook_points():
            hp.clear_context()
    
    def reset_hooks(self, clear_contexts=True, direction="both"):
        '''
        Reset hooks. 
        '''
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction)
        self.is_caching = False
    
    def add_hook(self, name, hook, dir="fwd"):
        '''
        Add hooks.
        '''
        if type(name) == str:
            self.mod_dict[name].add_hook(hook, dir=dir) # Add hook in the forward pass to the name key if name is a string
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in self.hook_dict.items():
                if name(hook_name):
                    hp.add_hook(hook,dir=dir)
    
    def run_with_hooks(
            self, 
            *model_args,
            fwd_hooks:List=[],
            bwd_hooks:List=[],
            reset_hooks_start:bool = True,
            reset_hooks_end:bool = True,
            clear_contexts:bool=False,
            **model_kwargs,
    )-> Tensor:
        '''
        Run cache with hooks.If we want to use backward hooks, we need to set reset_hooks_end to be False.
        Args:
            model_args and model_kwargs: All positional arguments and keyword arguments not otherwise captured are inputs to the model.
            fwd_hooks(List): A list of (name, hook), where name is either the name of a hook point or a Boolean funciton on hook names.
            Hook is the function to add to that hook point, or the hook whose names evaluate to True respectively. 
            reset_hooks_start (bool): A boolean of whether all prior hooks are removed at the start or not.
            reset_hooks_end (bool): A boolean of whether all prior hooks are removed at the end or not. 
            clear_contexts (bool): A boolean if whether hook contexts are cleared whenever hooks are reset.
        Returns:
            Tensor

        '''
        if reset_hooks_start: # If prior hooks are reset at the start
            if self.is_caching: # If we are caching
                logging.warning("Caching is on, but hooks are being reset") # Warning
            self.reset_hooks(clear_contexts) # Clear context
        
        for name, hook in fwd_hooks: # Loop through name and hook in forward hooks
            if type(name) == str: # If name type is string
                self.mod_dict[name].add_hook(hook, dir="fwd") # Add hook to the forward pass of the module dictionary of the given name
            else: # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.hook_dict: # Loop through hook name and hook point in hook dictionary
                    if name(hook_name): # If hook name is in name
                        hp.add_hook(hook,dir="bwd") # Add hook to the hook points in the backward pass

        out = self.forward(*model_args, **model_kwargs) 
        if reset_hooks_end: # If prior hooks are reset at the end
            if len(bwd_hooks) > 0:
                logging.warning(
                    "WARNING: Hooks were reset at the end of run_with_hooks while backward hooks were set. This removes the backward hooks before a backward pass can occur."
                )
            self.reset_hooks(clear_contexts) 
        return out  
    def add_caching_hooks(
            self,
            names_filter:Optional[Any]=None,
            incl_bwd:bool=False,
            device:Optional[str]=None,
            remove_batch_dim:bool=False,
            cache:Optional[dict]=None,
            verbose=False,
    )-> Dict:
        '''
        Adds hooks to the model to cache activations. It does not actually run the model to get activations, that must be done separately.
        Args:
            names_filter(NamesFilter, optional): a names filter of which activation to cache. Can be a list of strings (hook names) or a filter function mapping
            hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): A boolean of whether to also use backwards hooks. Defaults to False.
            device (__type__, optional): The device to store on. Defaults to CUDA if available else CPU.
            remove_batch_dim(bool, optional): A boolean of whether to remove the batch dimension (only works for batch size of 1). Defaults to False.
            cache (Optional[dict], optional): Cache to store activations in, a new dict is created by default. Defaults to None. 
        Returns:
            Cache (dict): Cache where activations will be stored. 

        '''
        if remove_batch_dim: # If we remove the batch dimension
            logging.warning(
                "Remove batch dim is caching hooks is deprecated. Use the Cache object or run_with_cache flags instead."
            )
        if device is None: # If no device
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if cache is None: # If no cache
            cache = {}
        if names_filter is None: # If no names filter
            names_filter = lambda name: True # Set lambda: name to True
        elif type(names_filter) == str: # If type of names_filter is string
            names_filter = lambda name: name == names_filter # names filter will be name given that name is names filter
        elif type(names_filter) == list: # If type of names_filter is list
            names_filter = lambda name: name in names_filter # names filter will be name for name in names filter 
        
        self.is_caching = True # We are caching

        def save_hook(
                    tensor: Tensor,
                    hook: HookPoint,
                    ) -> None:
            '''
            Save hooks.
            Args:
                tensor(Tensor): A tensor.
                hook (Tensor): A tensor of hook.
                verbose(bool): A boolean of whether to print certain statements or not.
            Returns:
                None
            '''
            if verbose:
                print("Saving  ", hook.name)
            if remove_batch_dim: 
                cache[hook.name] = tensor.detach().to(device).clone()[0] # Assign the value of name of hook in cache to a clone of the first dimension of tensor (Tensor)
            else:
                cache[hook.name] = tensor.detach().to(device).clone() # Assign the value of name of hook in cache to a clone of tensor

        def save_hook_back(self,
                           tensor: Tensor,
                           hook: HookPoint) -> None:
            '''
            Save hooks in the backward pass.
            Args:
                tensor(Tensor): A tensor.
                hook (Tensor): A tensor of hook.
                verbose(bool): A boolean of whether to print certain statements or not.
            Returns:
                None
            '''
            if verbose:
                print("Saving ", hook.name)
                if remove_batch_dim:
                    cache[hook.name + "_grad"] = tensor[0].detach().clone().to(device)[0] # If we remove batch dimension, the value of hook name + "_grad" is assigned to a clone of the gradient of tensor
                else: 
                    cache[hook.name + "_grad"] = tensor[0].detach().clone().to(device) # Otherwise, the value of hook name + "_grad" is assigned to the first dimension of a clone of a tensor

        for name, hp in self.hook_dict.items(): # Loop through name and hook point in hook dictionary
            if names_filter(name): # If filter by name
                hp.add_hook(save_hook,"fwd") # Add hooks saved in the forward pass
                if incl_bwd: # If include backward pass
                    hp.add_hook(save_hook_back,"bwd") # Add hooks saved in the backward pass 

        return cache
    
    #def run_with_cache(
    #        self, 
    #        *model_args,
    #        names_filter:NamesFilter=None,
    #        device:Optional[str]=None,
    #        remove_batch_dim:bool=False,
    #        incl_bwd:bool=False,
    #        reset_hooks_end:bool=True,
    #        reset_hooks_start:bool=True,
    #        clear_context:bool=False,
    #        return_cache_object:bool=True,
    #        **model_kwargs,
    #):
    #    '''
    #    Runs the model and returns model output and a Cache object.
    #    Args:
    #        model_args and model_kwargs: All positional arguments and keyword arguments not otherwise captured are inputs to the model. 
    #        names_filter(None or str or [str] or fn:str->bool): A filter for which activations to cache. Defaults to None, meaning cache everything.
    #        device (str or torch.Device): The device to cache activations on, defaults to model device. This must be set if the mode does not have a model.cfg.device attribute. 
    #        remove_batch_dim(bool): A boolean of whether to remove the batch dimension or not (Only works for batch size of 1).
    #        incl_bwd(bool) 
    #    '''



        




        
    

