from typing import List, Optional, Any, Tuple, Dict, Union
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
from torch import Tensor
import numpy as np

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


def get_circuit_replacement_hook(
        heads_to_remove: Optional[Dict]= None, 
        mlps_to_remove: Optional[Dict] = None,
        heads_to_keep: Optional[Dict] = None,
        mlps_to_keep:Optional[Dict] = None,
        heads_to_remove2: Optional[Dict] = None,
        mlps_to_remove2: Optional[Dict] = None,
        heads_to_keep2: Optional[Dict]=None,
        mlps_to_keep2: Optional[Dict]=None,
        dataset:Optional[Any]=None,
        model:Optional[nn.Module]=None,
):
    '''
    Obtains the circuit replacement hook.
    Args:
        heads_to_remove(Optional[Dict]): An optional dictionary of heads to remove.
        mlps_to_remove(Optional[Dict]): An optional dictionary of MLPS to remove.
        heads_to_keep (Optional[Dict]): An optional dictionary of heads to keep. 
        mlps_to_keep (Optional[Dict]): An optional dictionary of MLPs to keep.
        heads_to_remove2 (Optional[Dict]): An optional second dictionary of heads to remove.
        mlps_to_remove2 (Optional[Dict]): An optional second dictionary of MLPs to remove.
        heads_to_keep2 (Optional[Dict]): An optional second dictionary of heads to keep.
        mlps_to_keep2 (Optional[Dict]): An optional second dictionary of MLPs to keep.
        dataset (Optional[Any]): An optional dataset to take samples from.
        model (Optional[nn.Module]): An optional model to remove and keep heads and MLPs from.
    Returns:
        Tuple of circuit_replmt_hook, heads and mlps
    '''
    assert model is not None, "model cannot be None"
    assert dataset is not None, "dataset cannot be None"
    heads, mlps = process_heads_and_mlps(
        heads_to_remove=heads_to_remove, # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
        mlps_to_remove=mlps_to_remove2, # {2: List[List[int]]: dimensions dataset_size * datapoint_length
        heads_to_keep=heads_to_keep2, # as above for heads
        mlps_to_keep=mlps_to_keep, # as above for mlps
        dataset=dataset,
        model=model,
    )
    if (heads_to_remove2 is not None) or (heads_to_keep2 is not None):
        heads2, mlps2 = process_heads_and_mlps(
            heads_to_remove=heads_to_remove2,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
            mlps_to_remove=mlps_to_remove2, # {2: List[List[int]]: dimensions dataset_size * datapoint_length
            heads_to_keep=heads_to_keep2, # as above for heads
            mlps_to_keep=mlps_to_keep2, # as above for mlps
            dataset=dataset,
            model=model,
        )
    else: 
        heads2, mlps2 = heads, mlps

    
    dataset_length = dataset.N 

    if hasattr(dataset, "N"):
        dataset_length = int(getattr(dataset,"N")) # If dataset has attribute "N", assign dataset_length to the integer of N
    elif hasattr(dataset, "__len__"):
        dataset_length = len(dataset) # If dataset has attribute "__len__"(length), assign dataset_length to the length of dataset
    else:
        raise AttributeError("Dataset has no attribute 'N' and is not sized (no __len__).")
    
    def circuit_replmt_hook(z:Tensor,
                            act,
                             hook) -> Tensor:
        '''
        Obtains circuit replacement hooks.
        Args:
            z(Tensor): A tensor of inputs
            act: Activation
            hook (Tensor): A tensor of hook points
        Returns:
            A tensor of z
        '''
        layer = int(hook.name.split(".")[1]) # Obtain the layer
        # Create a range of indices for the first dimension
        batch_indices = torch.arange(dataset_length) 
        if "mlp" in hook.name and layer in mlps:
            # Extract the specific indices for the second dimension
            z_idx = mlps[layer]
            act_idx = mlps2[layer]

            # Vectorisation
            z[batch_indices, z_idx, :] = act[batch_indices, act_idx, :]
             # Ablate all the indices in mlps[layer] with batch size of data_length; mean may cntain semantic ablation

        if "attn.hook_result" in hook.name and (layer, hook.ctx["idx"]) in heads:
            heads_idx = heads[(layer, hook.ctx["idx"])]
            heads2_idx = heads2[layer,hook.ctx["idx"]]
            z[batch_indices, heads_idx,:] = act[batch_indices, heads2_idx, :]

        return z
    
    return circuit_replmt_hook, heads, mlps

def join_lists_nested(
        l1:List[list], l2:List[int],
)-> Tensor:
    '''
    Joins two list together. 
    Args:
        l1(List[list]): A list of list.
        l2(List(int)): A list of integers.
    Returns:
        A list of list and integers.
    '''
    rows = list(torch.nested.nested_tensor(l1).unbind()) # Unbind the nested structure into a list of separate tensors for speed
    l2_tensor = torch.tensor(l2)
    updated_list = [torch.cat([rows[i], l2_tensor[i].unsqueeze(0)]) for i in range(len(rows))]
    
    # Rebind into a single NestedTensor
    return torch.nested.as_nested_tensor(updated_list)


def get_extracted_idx(idx_list:List[str], dataset: Any) -> Tensor:
    '''
    Obtains extracted index from a list of indices. 
    Args:
        idx_list(List[str]): A list of indices
        dataset(Any): A dataset
    Returns:
        an integer of index
    '''
    int_idx = [[] for i in range(len(dataset.sentences))]
    # Get all integer IDs in one pass
    for idx_name in idx_list:
        try:
            int_idx_to_add = [
                int(x) for x in list(dataset.word_idx[idx_name])
            ]
        except:
            print(dataset.word_idx, idx_name)
            raise ValueError(
                f"Index {idx_name} not found in the dataset. Please check the spelling and ensure the index is in the dataset."
            )
    int_idx_result = join_lists_nested(int_idx,int_idx_to_add)
        #int_idx = [list(int_idx_result[i].tolist()) for i in range(len(int_idx_result))]
    return int_idx_result

def do_circuit_extraction(
        heads_to_remove: Optional[Dict]=None,
        mlps_to_remove: Optional[Dict]=None,
        heads_to_keep: Optional[Dict]=None,
        mlps_to_keep: Optional[Dict]=None,
        dataset:Optional[Any]=None,
        mean_dataset: Optional[Any]=None,
        model:Optional[Any]=None,
        metric:Optional[str]=None,
        excluded:Optional[List]=[],
        return_hooks:bool=False,
        hooks_dict:bool=False,
)-> Tuple[Any,Tensor]:
    '''
    Extracts circuits and performs ablation. 
    Args:
        heads_to_remove (Optional[Dict]): An optional dictionary of heads to remove. Defaults to None.
        mlps_to_remove (Optional[Dict]): An optional dictionary of mlps to remove. Defaults to None.
        heads_to_keep (Optional[Dict]): An optional dictionary of heads to keep. Defaults to None.
        mlps_to_keep (Optional[Dict]): An optional dictionary of mlps to keep. Defaults to None.
        dataset (Optional[Any]): An optional dataset. Defaults to None.
        mean_dataset (Optional[Any]): An optional mean dataset. Defaults to None.
        model (Optional[Any]): An optional model. Defaults to None.
        metric (Optional[str]): A optional string of a metric. Defaults to None.
        excluded (Optional[List]): An optional list of excluded heads that we do not put any hooks on. Defaults to an empty list.
        return_hooks (bool): A boolean of whether to return hooks or not. Defaults to False.
        hooks_dict (bool): A boolean of whether to put hook in a dictionary or not. Defaults to False.
    Returns:
        A Tuple of a model and a tensor of ablation 
    '''

    # Check whether we are either in keep XOR remove moved from the args
    ablation, heads, mlps = get_circuit_replacement_hook(
        heads_to_remove=heads_to_remove,
        mlps_to_remove=mlps_to_remove,
        heads_to_keep=heads_to_keep,
        mlps_to_keep=mlps_to_keep,
        dataset=dataset,
        model=model
    )
    metric = ExperimentMetric(
        metric=metric, dataset=dataset.input, relative_metric=False
    )





    


    


    