from copy import deepcopy
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import torch
from vit_prisma.utils.experiments import get_act_hook
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict
from vit_prisma.utils.ioi_utils import show_pp
import graphviz 
from vit_prisma.utils import detect_architectures
from torch import Tensor

def get_hook_tuple(model:Any,
                   layer: int,
                   head_idx: int,
                   comp: str=None, 
                   input: bool=False,
                   n_layers: int=12) -> Tuple[str, Callable]:
    '''
    Helper function to create a hook tuple for a given layer and head index.
    Args:
        model (Any): The model for which to create the hook tuple.
        layer (int): The layer number for which to create the hook tuple.
        head_idx (int): The head index for which to create the hook tuple.
        comp (str, optional): The computation for which to create the hook tuple. Defaults to None.
        input (bool, optional): Whether to include the input in the hook tuple. Defaults to False.
        n_layers (int, optional): The total number of layers in the model. Defaults to 12.
    Returns:
        Tuple[str, Callable]: A tuple containing the hook name and the hook function.
    '''
    arch = detect_architectures(model)
    HOOKS = {}

    if comp in ["q", "k", "v"]:
        hook_pattern = HOOKS[arch][comp]
        assert comp in ["q", "k", "v"], f"Invalid comp: {comp}. Must be one of 'q', 'k', or 'v'."
        assert head_idx is not None, f"head_idx must be provided when comp is one of 'q', 'k', or 'v'."
        return hook_pattern.format(layer=layer, head_idx=head_idx)
    elif head_idx is None:
        if layer < n_layers:    
            hook_pattern = HOOKS[arch]["mlp_out" if not input else "resid_mid"] 
        else:
            hook_pattern = HOOKS[arch]["resid_post"]
        return hook_pattern.format(layer=layer, head_idx=head_idx)

def patch_all(z: Tensor,
              source_act: Tensor,
               hook: Tensor,) -> Tensor:
    '''
    Patch the activations of a model with the activations from a source model.
    Args:
        z (Tensor): The activations of the model to be patched.
        source_act (Tensor): The activations from the source model to patch with.
        hook (Tensor): The hook tensor indicating where to patch.
        Returns:
        Tensor: The patched activations.
    '''
    z[:] = source_act[hook]
    return z

def patch_positions(
        z: Tensor,
        source_act: Tensor,
        hook: Tensor,
        positions: List[int],
) -> Tensor:
    '''
    Patch the activations of a model at specific positions with the activations from a source model.
    Args:
        z (Tensor): The activations of the model to be patched.
        source_act (Tensor): The activations from the source model to patch with.
        hook (Tensor): The hook tensor indicating where to patch.
        positions (List[int]): The list of positions to patch.
    Returns:
        Tensor: The patched activations.
    '''
    if positions is None: # Same as patch all
        raise NotImplementedError(
            "Patching all positions is not implemented in patch_positions. Use patch_all instead."
        )
    else:
        batch = z.shape[0]
        cur_positions = Tensor(positions)
        if len(cur_positions.shape) == 0:
            cur_positions = cur_positions.unsqueeze(0)
        for pos in cur_positions: 
            z[torch.arange(batch), pos] = source_act[torch.arange(batch), pos]
        return z

def get_datasets(text: str,
                 dataset: Any) -> Tuple[Tensor, Tensor]:
    '''
    Get the datasets for the circuit discovery experiments.
    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input and output datasets.
    '''
    batch_size = 1
    orig = text
    output = dataset(orig, return_tensors="pt", padding=True)
    obj = [obj for obj in dataset if obj != output]

    return orig, output,obj

def old_path_patching(
        model: Any,
        orig_data: Tensor,
        new_data: Tensor,
        senders: List[Tuple],
        receiver_hooks: List[Tuple],
        max_layer: Union[int, None] = None,
        position: int = 0,
        return_hooks: bool = False,
        freeze_mlps: bool = True,
        orgin_cache: Optional[Dict] = None,
        new_cache: Optional[Dict] = None,
        prepend_bos: bool = False,

) -> Union[Tensor, Tuple[Tensor, Dict, Dict]]:
    '''
    Perform path patching on a model by patching the activations from a source model to a target model in the old way.
    Args:
        model (Any): The model to perform path patching on.
        orig_data (Tensor): The original input data for the model.
        new_data (Tensor): The new input data for the model.
        senders (List[Tuple]): A list of tuples containing the sender layer and head index for patching.
        receiver_hooks (List[Tuple]): A list of tuples containing the receiver layer and head index for patching.
        max_layer (Union[int, None], optional): The maximum layer to consider for patching.
        position (int, optional): The position to patch. Defaults to 0.
        return_hooks (bool, optional): Whether to return the hooks used for patching. Defaults to False.
        freeze_mlps (bool, optional): Whether to freeze the MLPs during patching. Defaults to True.
        orgin_cache (Optional[Dict], optional): The cache for the original model. Defaults to None.
        new_cache (Optional[Dict], optional): The cache for the new model. Defaults to None.
        prepend_bos (bool, optional): Whether to prepend a beginning of sequence token to the input data. Defaults to False.
    '''
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers, f"max_layer must be less than or equal to the number of layers in the model ({model.cfg.n_layers})."

    if orig_cache is None:

        # Save activations from original data
        orig_cache = {}
        model.reset_hooks()
        model.cache_all (orig_cache)
        _ = model(orig_data, prepend_bos=False)

    # Process senders
    senders_hooks = []
    for layer, head in senders:
        if head is not None: #(layer, head) for attention heads
            senders_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else: # (layer, None) for MLPs
            senders_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in senders_hooks]

    if new_cache is None:
        # Save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in sender_hook_names)
        _ = model(new_data, prepend_bos=False)

    else:
        assert all(
            [x in new_cache for x in sender_hook_names]
        ), f"Difference between new_cahce and senders: {set(sender_hook_names) - set(new_cache.keys())}"

    # Configure hooks for freezing activations
    for layer in range(max_layer):
        # heads
        for head in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)
                hook = get_act_hook(
                    patch_all,
                    alt_act=orig_cache[hook_name],
                    idx=head,
                    dim=2,
                )
                model.add_hook(hook_name, hook)
    
    # MLPs
    if freeze_mlps:
        hook_name = f"blocks.{layer}.hook_mlp_out"
        hook = get_act_hook(
            patch_all,
            alt_act=orig_cache[hook_name],
            idx=None,
            dim=None,
        )
        model.add_hook(hook_name, hook)
    
    # For senders, add new hook to patching new activations
    for hook_name, head in senders_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=new_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        model.add_hook(hook_name, hook)

    # Forward pass on origin, where patch in new activations for senders and original activations for all other hooks (including receiver_hooks). 
    # Save activations on receivers

    _ = model(orig_data, prepend_bos=False)
    model.reset_hooks()

    # Add hooks for final forward pass on origin data, where we patch in hybrid activations for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=orig_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        hooks.append((hook_name, hook))

    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model
    
#def direct_path_patching(
#        model: Any,
#        orig_data: Tensor,
#        new_data: Tensor,
#)



        
        

        
    
