"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Sequence
from vit_prisma.prisma_tools.lens_handle import LensHandle
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

        self.ctx = {} # what is this?

        self.name = None
    
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
            fwd_hooks=[],
            bwd_hooks=[],
    ):

        
    

