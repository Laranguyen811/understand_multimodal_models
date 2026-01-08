from typing import List, Optional, Any, Tuple, Dict
import warnings
from copy import deepcopy
from vit_prisma.utils.experiments import (ExperimentMetric,
AblationConfig,
Ablation,
Patching,
PatchingConfig,
get_act_hook
)

import torch 
import plotly.express as px
import gc
import einops
from torch import nn


def list_diff(l1:list,l2:list)-> list:
    '''
    Calculates a list of difference between 2 lists.
    Args:
        l1: The first list.
        l2: The second list.
    Returns:
        A list of difference
    '''
    l2_ = [int(x) for x in l2]
    return list(set(l1).difference(set(l2_)))

def turn_keep_into_rmv(to_keep: Dict, max_len: Dict) -> Dict:
    '''
    Turn to_keep dictionary into to_rmv dictionary (to remove).
    Args:
        to_keep(dict): A dictionary of to_keep.
        max_len(int): An integer of max length.
    Returns:
        A dictionary of to_rmv.
    '''
    to_rmv = {}
    for t in to_keep.keys():
        to_rmv[t] = []
        for idxs in to_keep[t]:
            to_rmv[t].append(list_diff(list(range(max_len)),idxs))
        
    return to_rmv

def process_heads_and_mlps(
        heads_to_remove:Optional[Dict]=None, 
        mlps_to_remove:Optional[Dict]=None,
        heads_to_keep:Optional[Dict]=None,
        mlps_to_keep:Optional[Dict]=None,
        dataset:Optional[Any]=None,
        model:Optional[Any]=None,

) -> Tuple[dict,dict]:
    '''
    Processes heads and mlps to remove.
        Args:
        heads_to_remove: Dict with (layer, head) tuples as keys, 
                        values are List[List[int]] of shape (dataset_size, datapoint_length)
        mlps_to_remove: Dict with layer int as keys,
                       values are List[List[int]] of shape (dataset_size, datapoint_length)
        heads_to_keep: Dict with (layer, head) tuples as keys (alternative to heads_to_remove)
        mlps_to_keep: Dict with layer int as keys (alternative to mlps_to_remove)
        dataset: Dataset with .N and .max_len attributes
        model: Model with .cfg.n_layers and .cfg.n_heads attributes
    
    Returns:
        Tuple of (heads dict, mlps dict) in remove format

    '''
    assert (heads_to_remove is None) != (heads_to_keep is None) # Ensure that either heads_to_keep or heads_to_remove is not empty 
    assert (mlps_to_keep is None) != (mlps_to_remove is None) # Ensure that either mlps_to_keep or mlps_to_remove is not empty
    assert model is not None, "model cannot be None"
    assert dataset is not None, "dataset cannot be None"
    
    heads: dict = {}
    mlps: dict = {}
    
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    dataset_length = dataset.N
    # Ensure dataset is provided and determin its length safely
    if dataset is None:
        raise AssertionError("Dataset cannot be None.")
    
    if hasattr(dataset, "N"):
        dataset_length = int(getattr(dataset,"N")) # If dataset has attribute "N", assign dataset_length to the integer of N
    elif hasattr(dataset, "__len__"):
        dataset_length = len(dataset) # If dataset has attribute "__len__"(length), assign dataset_length to the length of dataset
    else:
        raise AttributeError("Dataset has no attribute 'N' and is not sized (no __len__).")
    
    if mlps_to_remove is not None:
        mlps = mlps_to_remove.copy()
    elif mlps_to_keep is not None:
        mlps = mlps_to_keep.copy()
        for l in range(n_layers):
            if mlps_to_keep is not None and l not in mlps_to_keep:
                mlps[l] = [[] for _ in range(dataset_length)] # Leave the list of l empty 
        mlps = turn_keep_into_rmv(mlps, dataset.max_len)

    if heads_to_remove is not None:
        heads = heads_to_remove.copy()
    elif heads_to_keep is not None:
        heads = heads_to_keep.copy()
        
        for l in range(n_layers):
            for h in range(n_heads):
                if (l,h) not in heads_to_keep:
                    heads[(l,h)] = [[] for _ in range(dataset_length)]
        
        heads = turn_keep_into_rmv(heads, dataset.max_len)
    return heads, mlps


