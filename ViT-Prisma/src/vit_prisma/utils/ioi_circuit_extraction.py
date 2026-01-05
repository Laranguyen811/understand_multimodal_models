from typing import List, Optional
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

def list_diff(l1: list,l2: list)-> list:
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

def turn_keep_into_rmv(to_keep: dict, max_len:int) -> dict:
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
        heads_to_remove:Optional[List[List[int]]]=None, 
        mlps_to_remove:Optional[List[List[int]]]=None,
        heads_to_keep:Optional[List[List[int]]]=None,
        mlps_to_keep:Optional[List[List[int]]]=None,
        dataset=
)
