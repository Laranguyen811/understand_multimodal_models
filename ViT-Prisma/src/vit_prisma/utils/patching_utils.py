
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from rich.progress import track
from vit_prisma import utils
import numpy as np
from torch import Tensor
from jaxtyping import Bool, Float
from typing import Any, List, Tuple, Optional, Dict
from rich.progress import track
from vit_prisma.prisma_tools import activation_cache
import warnings
from vit_prisma.utils.experiments import get_act_hook
from vit_prisma.utils.detect_architectures import detect_architecture

def calculate_logits_to_ave_logit_diff(
        logits: Tensor,
        test_case: dict,
        per_prompt: Bool = False,
    ) -> Float[Tensor, "batch"]:
    '''
    Returns the average logit difference between the correct and distractor prompts.
    Args:
        logits(Tensor): Logits tensor of shape (batch, seq, d_model).
        per_prompt (Bool): If True, returns the logit difference per prompt.
    Returns:
        Average logit difference tensor of shape (batch,).
    '''
    # Calculate the logit difference between correct and distractor rounded to 3 decimal places
    logits_diff = np.round((logits[:, test_case['correct_idx']] - logits[:, test_case['distractor_idx']]).item(),3)
    # If per_prompt is True, return the logit difference per prompt
    
    # Calculate mean logit difference rounded to 3 decimal places
    mean_logits_diff = np.round(logits_diff.mean().item(),3)
    
    return logits_diff if per_prompt else mean_logits_diff

class ObjectData(Dataset):
    '''
    A custom dataset class for handling object data.
    '''
    def __init__(self,data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
def patch_head_vector_at_pos(
        clean_head_vector: Tensor,
        hook,
        head_index: int,
        pos_index: int,
        corrupted_cache,
    
):
    clean_head_vector[:, pos_index, head_index, :] = corrupted_cache[hook.name][:, pos_index, head_index, :]
    return clean_head_vector

def cache_activation_hook(
        activation: Tensor,
        hook,
        my_cache={}
):
    '''
    Storing activation.
    Args:   
        hook(Tensor): A tensor of a hook point. 
        my_cache(ActivationCache): An ActivationCache of a cache to obtain activation
    Returns:
        Tensor
    '''
    my_cache[hook.name] = activation

def patch_full_residual_component(
        corrupted_residual_component: Float[Tensor, "batch pos d_model"],
        hook,
        pos_index: int,
        corrupted_cache):
    '''
    Patching with corrupted residual component.
    Args:
        corrupted_residual_component(float): A float of a residual component to patch with.
        hook(Tensor): A tensor of a hook point.
        pos_index (int): An integer of a positional index.
        corrupted_cache (ActivationCache): An ActivationCache of a cache that has been corrupted. 
    Return:
        Tensor
    '''
    corrupted_residual_component[:, pos_index, :] = corrupted_cache[hook.name][:, pos_index, :]
    return corrupted_residual_component

def patch_all(z: Tensor, source_act: Tensor, hook: Tensor):
    '''
    Patch the source activations.
    Args:
        z(Tensor): A tensor of a corrupted activation.
        source_act (Tensor): A source activation.
        hook (Tensor): A tensor of a hook point. 
    Returns:
        Tensor
    '''
    return source_act

def get_hook_tuple(
        model: Any,
        layer: int, 
        head_idx: int, 
        comp=None, 
        input:Bool=False,
        n_layers: int=12,
        ) -> Tuple:
    '''
    Gets the hook tuple.
    Args:
        layer (int): An integer of the layer.
        head_idx (int): An integer of the head index.
        comp (str): A string of the hook name, set to None.
        input (Bool): A boolean of whether there is an input or not, set to None.
    Returns:
        Tuple of (hook_name, head_idx)
    '''
    arch = detect_architecture(model)
    HOOKS = {}

    # Determin component 
    if comp in ['q','k','v']:
        hook_pattern = HOOKS[comp][arch]
    elif head_idx is None:
        hook_pattern = HOOKS[arch]['mlp_out' if not input else 'resid_mid']


                


def path_patching(
        model: Any,
        D_new,
        D_orig,
        sender_heads,
        receiver_hooks,
        positions=["end"],
        return_hooks:Bool=False,
        extra_hooks=[],
        freeze_mlps: Bool=False,
        have_internal_interactions: Bool = False,
                  ):
    '''
    Isolates indirect effects from direct effects by replacing components in a model's forward pass with different activations and comparing the change.
    
    Args:
        model: Model to perform path patching on.
        D_orig: The original data point.
        D_new: The new data point.
        sender_heads: The sender attention head.
        receiver_hooks: The receiver hooks (Hooks allow us to do something after an interpretability frameworks compute intermediate values).
        positions: Positions to patch.
        return_hooks: Whether to return hooks or not. 
        extra_hooks: Store extra hooks.
        freeze_mlps: Whether to freeze MLPs (Multiple Layer Perceptrons) or not.
        have_internal_interactions: Whether there are internal interactions or not. 
    
    Returns:
        Tensor or Model
    '''
    def patch_positions(z, source_act, hook, positions["end"], verbose=False):
        '''
        Determines the positions to patch. 
        '''
        for pos in positions:
            z[torch.arange(D_orig.N), D_orig.word_idx[pos]] = source_act[torch.arange(D_new.N), D_new.word_idx[pos]] # Creates a new tensor

        return z

    # Process arguments

    sender_hooks = []
    for layer, head_idx in sender_heads:
        if head_idx is None: # If head index is not empty
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out",None)) # Append to sender hooks

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    # Forward pass A
    sender_cache = {}
    model.reset_hooks() # Reset hooks in model
    for hook in extra_hooks: # Looping through extra hooks
        model.add_hook(*hook) # Add hook to model
    model.cache_some(
        sender_cache, lambda x:x in sender_hook_names, suppress_warning=True
    ) # Cache the activations from certain specified locations into the sender_cache dictionary, and only adds activations from hook points whose names are in the sender_hook_names 

    # Forward pass B
    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hooks(*hook)
    model.cache_all(target_cache, suppress_warning=True) # Cache all the target cache
    target_logits = model(D_orig.toks.long())

    # Forward pass C
    # Cache the receiver hooks
    # Adding these hooks first means we save values before they are overwritten
    receiver_cache = {}
    model.reset_hooks()
    model.cache_some(
        receiver_cache,
        lambda x: x in receiver_hook_names,
        suppress_warning=True,
        verbase=False
    ) # Cache activations from receiver. Only those in receiver_hook_names will be cached

# Freeze intermediate heads to their D_orig values
    for layer in range(model.cf.n_layers): # Looping through layers
        for head_idx in range(model.cfg.n_heads): # Looping through number of heads
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if have_internal_interactions and hook_name in receiver_hook_names:
                    continue # Continue if there are internal interactions and hook name is in receiver_hook_names

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                    )
                model.add_hook(hook_name, hook)
            
            if freeze_mlps:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=None,
                    dim=None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        for hook in extra_hooks:
            model.add_hook(*hook) # Adding hook in extra_hooks to model


        # These hooks will overwrite the freezing, for the sender heads
        for hook_name, head_idx in sender_hooks:
            assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]),(
                hook_name,
                head_idx,
            ) # Check if sender cache is similar to target cache for hook name and head index in sender_hooks

            hook = get_act_hook(
                partial(patch_positions, positions=positions), # Create a new function
                alt_act=sender_cache[hook_name], # Use the activation
                idx=head_idx, # The index is the head index
                dim=2 if head_idx is not None else None, # Dimension is 2 if head index is not empty; otherwise, None.
                name=hook_name # Name is hook name
            )
            model.add_hook(hook_name,hook)
            receiver_logits = model(D_orig.toks.long()) # Assign receiver logits to the logits of the original data point

        # Add (or return) all the hooks needed for forward pass D
        model.reset_hooks()
        hooks = []
        for hook in extra_hooks:
            hook.append(hook)
        
        for hook_name, head_idx in receiver_hooks: # Loop through hook names and head indices in receiver hooks
            for pos in positions: # Loop through position in positions
                if torch.allclose(receiver_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],
                                  target_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],): # If the elements of receiver cache and target cache at hook name and the position of word index and D_orig.N are all close (numerically similar)
                    
                    warnings.warn("Torch all close for {}".format(hook_name))
            
            hook = get_act_hook(
                partial(patch_positions, positions=positions), # Create a new function
                alt_act=receiver_cache[hook_name], # Use the activation
                idx=head_idx, # The index is the head index
                dim=2 if head_idx is not None else None, # Dimension is 2 if head index is not empty; otherwise, None. 
                name=hook_name, # Name is hook name
            ) # Get the activation hook
            hooks.append((hook_name, hook)) # Append hook name and hook to hooks
        
        model.reset_hooks()
        if return_hooks:
            return hooks
        else:
            for hook_name, hook in hooks:
                model.add_hook(hook_name, hook)
            return model

def direct_path_patching(
        model: Any,
        orig_data: Tensor,
        new_data: Tensor,
        initial_receivers_to_senders: List[Tuple[Tuple[str, Optional[int]], Tuple[int, Optional[int], str]]],
        orig_positions: int,
        new_positions: int,
        orig_cache = None,
        new_cache = None,


) -> Any:
    '''
    Path patching for direct effects only, excluding indirect effects.
    Args:
        model: A model to path patch.
        orig_data: An original data point.
        new_data: A new data point. 
        initial_receivers_to_senders: Edges patched from new cache.
        orig_positions: Positions of the original data point.
        new_positions: Positions of the new data point.
        new_cache: Cache of the new data point.
    Returns:
        Model
    '''
    # Caching
    if orig_cache is None:
        # Save activations from the original data point
        model.reset_hooks()
        orig_cache = {}
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False) # A placeholder variable 
        model.reset_hooks() # Reset hooks

    initial_senders_hook_names = [
        get_hook_tuple(item[1][0], item[1][1])[0]
        for item in initial_receivers_to_senders
    ] 
    if new_cache is None:
        # Save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in initial_senders_hook_names)
        _ = model(new_data, prepend_bos=False)
    else:
        assert all(
            [x in new_cache for x in initial_senders_hook_names]
        ), f"Incomplete new_cache. Missing {set(initial_senders_hook_names) - set(new_cache.keys())}"
    model.reset_hooks()

    # Set up a way for model components to dynamically see activations from the same forward pass
    for name, hp in model.hook_dict.items():
        assert (
            "model" not in hp.ctx or hp.ctx["model"] is model
        ), "Multiple models used as hook point references!"
        hp.ctx["model"] = model
        hp.ctx["hook_bname"] = name
    
    model.cache = (
        {}
    ) # This cache is populated and used on the same forward pass

