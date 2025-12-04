
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch as t
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

def path_patching(model: Any,
                  receiver_nodes,
                  source_tokens, 
                  patch_tokens,
                  ans_tokens, 
                  component: str = 'z',
                  position:int = -1,
                  freeze_mlps: Bool = False,
                  indirect_patch:Bool = False,
                  truncate_to_max_layer: Bool = False, 
                  ):
    '''
    Performs path patching.
    Args:
        model: Model to perform path patching on
        receiver_nodes: Terms in the forward pass that receive patching
        source_tokens: Tokens that are from source
        patch_tokens: Tokens to patch
        ans_tokens: Answer tokens (correct ones)
        component: Part of model we need to reverse engineer
        position: Position to path patch
        freeze_mlps: Whether to freeze (removing the influence of every path including one or more intermediate attention heads) MLPs or not 
        indirect_patch: Patch indirectly or not
        truncate_to_max_layer: Whether to truncate to max layer or not
    Returns:
        patched_head_pq_diff: Tensor
    '''
    model.reset_hooks() # Reset hooks in the model
    print(f"Component:{component}")

    # Obtain original logits and cache 
    original_logits, cache = model.run_with_cache(source_tokens)

    # Calculate the logit difference
    original_logit_diff = calculate_logits_to_ave_logit_diff(original_logits,ans_tokens)

    # Obtain label tokens
    label_tokens = ans_tokens[:, 0]

    # Obtain the original label logits
    original_label_logits = original_logits[:, -1][list(range(len(original_logits))), label_tokens]

    # Obtain corrupted logits and corrupted cache
    corrupted_logits, corrupted_cache = model.run_with_cache(patch_tokens)
    corrupted_logit_diff = calculate_logits_to_ave_logit_diff(corrupted_logits,ans_tokens)
    print(f"Corrupted logit difference:{corrupted_logit_diff}, Original logit difference: {original_logits}")

    del corrupted_logit_diff

    patched_head_pq_diff = t.zeros(model.cfg.n_layers, model.cfg.n_heads)


    def add_hook_to_attn(attn_block, hook_fn):
        if component=='v':
            attn_block.hook_v.add_hook(hook_fn)
        elif component=="q":
            attn_block.hook_q.add_hook(hook_fn)
        elif component=="k":
            attn_block.hook_k.add_hook(hook_fn)
        elif component=="z":
            attn_block.hook_z.add_hook(hook_fn)
        else:
            raise Exception(f"Component must be q,k,v, or z. You passed {component}")
        
    max_layer = model.cfg.n_layers
    if truncate_to_max_layer:
        target_layers = [r[0] for r in receiver_nodes]
        for t in target_layers:
            if type(t)==int: # If t is an integer
                max_layer = min(t, max_layer)
        if max_layer < model.cfg.n_layers:
            max_layer += 1 # Go up to max layer inclusive

    for layer in track(list(range(max_layer))):
        # Update progress information on each iteration
        for head_idx in range(model.cfg.n_heads):
            model.reset_hooks()
            if (layer, head_idx) in receiver_nodes:
                continue
                
            # Adding this before lets us cache the values before overwriting them
            receiver_cache = {}
            for recv_layer, recv_head in receiver_nodes:
                cache_fn = partial(cache_activation_hook, my_cache=receiver_cache)
                # Create a new function based on a pre-filled function 
                if recv_head is None:
                    model.add_hook(recv_layer, cache_fn)
                else: 
                    add_hook_to_attn(model.blocks[recv_layer].attn, cache_fn)
                
            # Add hooks for the sender nodes layer, head index
            hook_fn = partial(patch_head_vector_at_pos, head_idx=head_idx, pos_idx = position, )




    
        
    
