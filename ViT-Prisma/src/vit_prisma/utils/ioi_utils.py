from contextlib import suppress
import warnings
import plotly.graph_objects as go
import numpy as np
from numpy import sin, cos, pi
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from vit_prisma.utils.experiments import get_act_hook
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
import plotly.io as pio
from torch.utils.data import DataLoader
from functools import *
import gc
import collections
import copy
import itertools
from functools import partial
import numpy as np
from tqdm import tqdm
import pandas as pd
from IPython.core.getipython import get_ipython
from copy import deepcopy
from transformers import AutoModelForObjectDetection,AutoConfig, AutoTokenizer
from vit_prisma.prisma_tools.hook_point import HookedRootModule, HookPoint
from torch import Tensor

ALL_COLORS = px.colors.qualitative.Dark2
CLASS_COLORS = {
    "region mover": ALL_COLORS[0],
    "negative": ALL_COLORS[1],
    "distractor inhibition": ALL_COLORS[2],
    "induction": ALL_COLORS[5],
    "relationship binding": ALL_COLORS[3],
    "previous patch": ALL_COLORS[6],
    "none": ALL_COLORS[7],
    "back up region mover": "rgb(27,100,119)",
    "light back up region mover":"rgb(146,183,210)", 

 }

from vit_prisma.utils.ioi_circuit_extraction import get_extracted_idx

def clear_gpu_mem():
    """
    Clear GPU memory.
    """
    gc.collect()
    torch.cuda.empty_cache()

def show_tokens(tokens: Tensor, model: Any, return_list:bool=False) -> List|None: 
    '''
    Tokenises strings. 
    Args:
        token (Tensor): A tensor of token.
        model (Any): A model.
        return_list (bool): A boolean of whether to return a list or not.
    Returns:
        A list of tokens
    '''
    if type(tokens) == str:
        # If we input texts, tokenise first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    if return_list:
        return text_tokens
    else: 
        print("|".join(text_tokens)) # Print text tokens with "|" in between

    def max_2d(m: Tensor, k:int=1)-> Tuple[List,Tensor]:
        """
        Get the max of a matrix.
        Args:
            m (Tensor): An input of matrix tensor.
            k (int): An integer of top k.
        Returns:
            A tuple of List of out and Tensor of flattened matrix at specified indices. 

        """ 
        if len(m.shape) != 2:
            raise NotImplementedError()
        mf = m.flatten() # Flatten m
        inds = torch.topk(mf, k=k).indices
        out = []
        for ind in inds:
            ind = ind.item()
            x = ind // m.shape[1]
            y = ind - x* m.shape[1]
            out.append((x,y))
        return out, mf[inds]
    
    def show_pp(
            m: Tensor,
            xlabel:str="",
            ylabel:str="",
            title:str="",
            bartitle:str="",
            animate_axis=None,
            highlight_points=None,
            highlight_name:str="",
            return_fig:bool=False,
            show_fig:bool=True,
            **kwargs,
    ):
        """
        Plots a heatmap of the values in the matrix 'm'.
    
        """
        if animate_axis is None:
            fig = px.imshow(
                title=title if title else "",
                animation_frame=animate_axis,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                **kwargs,
            )
        else:
            fig = px.imshow(
                einops.rearrange(m, "a b c -> a c b"),
                title=title if title else "",
                animation_frame=animate_axis,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                **kwargs,
            )
        
        fig.update_layout(
            coloraxis_colorbar = dict(
                title=bartitle,
                thicknessmode="pixels",
                thickness = 50,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1, 
                ticks="outside",
            ),
        )

        if highlight_points is not None:
            fig.add_scatter(
                x=highlight_points[1],
                y=highlight_points[0],
                mode="markers",
                marker=dict(color="green",size=10, opacity=0.5),
                name=highlight_name,
            )


