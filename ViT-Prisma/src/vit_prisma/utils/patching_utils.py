
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from rich.progress import track
from vit_prisma import utils
import numpy as np
from torch import Tensor
from jaxtyping import Bool, Float
from typing import Any
from rich.progress import track
from vit_prisma.prisma_tools import activation_cache

def calculate_logits_to_ave_logit_diff(
        logits: Tensor,
        test_case: dict,
        per_prompt: Bool = False,
    ) -> Float[Tensor, "batch"]:
    '''
    Returns the average logit difference between the correct and distractor prompts.
    Args:
        logits: Logits tensor of shape (batch, seq, d_model).
        per_prompt: If True, returns the logit difference per prompt.
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
        hook: hook point
        my_cache: cache to obtain activation
    Returns:
        activation: Tensor
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
        corrupted_residual_component: residual component to patch with.
        hook: hook point
        pos_index: positional index
        corrupted_cache: cache that has been corrupted
    Return:
        corrupted_residual_component: Tensor
    '''
    corrupted_residual_component[:, pos_index, :] = corrupted_cache[hook.name][:, pos_index, :]
    return corrupted_residual_component

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
        receiver_hooks: The receiver hooks (Hooks allow us to do something after an interpretability frameworks compute intermediate values)
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
                partial(patch_positions, positions=positions),
                alt_act=sender_cache[hook_name],
                idx=head_idx,
                dim=2 if head_idx is not None else None, 
                name=hook_name
            )
            model.add_hook(hook_name,hook)
            receiver_logits = model(D_orig.toks.long())

