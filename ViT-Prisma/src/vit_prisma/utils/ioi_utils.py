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
from plotly.graph_objs import Figure
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
from vit_prisma.dataloaders.visual_genome import load_images, set_up_vg_paths, VisualGenomeDataset, transform
from vit_prisma.utils.data_utils.visual_genome.visual_genome_objects import create_dict, base_dir
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
loaded_images = load_images(train_path=set_up_vg_paths(verbose=True),batch_size=8,verbose=False)
print(f"Number of loaded images: {len(loaded_images)}")
objs,obj_dict = create_dict(base_dir=base_dir)
labels = obj_dict.values()
loaded_images_data = VisualGenomeDataset(loaded_images,labels=labels,transform=transform)
print(f"First image:{(loaded_images)}")
print(f"Type of loaded images:{type(loaded_images)}")

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

    fig.update_layout(
        yaxis_title = ylabel,
        xaxis_title = xlabel,
        xaxis_range = [-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
        
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig 
    

def show_attention_patterns(
        model: Any,
        heads: Dict[str,int],
        dataset: Any,
        outputs:Tensor,
        toks: Tensor,
        precomputed_cache=None,
        mode:str="val",
        title_suffix:str="",
        return_fig:bool=False,
        return_mtx:bool=False,
        
)-> int|Figure|None:
    '''
    Displays attention patterns
    Args:
        model (Any): A model
        heads (str): A list of strings or integers of attention heads.
        dataset(Any): A dataset.
        precomputed_cache: Precomputed cache. Defaults to None.
        mode (str): A string of mode. Defaults to to val.
        title_suffix(str): A string of title suffix. Defaults to "".
        return_fig(bool): A boolean of whether to return figure or not. Defaults to False.
        return_mtx(bool): A boolean of whether to return matrix or not. Defaults to False.
    Returns:
        An integer of attention results or a figure
    '''
    assert mode in [
        "attn",
        "val",
        "scores",
    ] # value weighted attention or attention for attention probabilities
    
    assert isinstance(
        dataset, objs
    ), f"Dataset must be Visual Genome {type(dataset)}."
    prompts = labels
    assert len(heads) == 1 or not (return_fig or return_mtx)

    for (layer, head) in heads:
        cache = {}
        good_names = [
            f"blocks.{layer}.attn.hook_attn" + ("_scores" if mode == "scores" else "") 
        ]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        if precomputed_cache is None:
            model.cache_some(
                cache=cache, names=lambda x: x in good_names
            ) 
            logits = model(outputs)
        else:
            cache = precomputed_cache
        attn_results = torch.zeros(
            size=(dataset.N, dataset.max_len, dataset.max_len)
        )
        attn_results += -20 

        for i, text in enumerate(prompts):
            toks = toks
            current_length = len(toks)
            inputs = [model.tokenizer.decode([tok]) for tok in toks]
            attn = cache[good_names[0]].detach().cpu()[i, head, :, :]

            if mode == "val":
                vals = cache[good_names[1]].detach().cpu()[i, :, head, :].norm(dim=-1)
                cont = torch.einsum("ab, b -> ab", attn, vals)
            
            fig = px.imshow(
                attn if mode in ["attn", "scores"] else cont,
                title=f"{layer}.{head} Attention" + title_suffix,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                labels={"y": "Queries", "x": "Keys"},
                height=500,
            )
            fig.update_layout(
                xaxis={
                    "side": "top",
                    "ticktext": inputs,
                    "tickvals": list(range(len(inputs))),
                    "tickfont": dict(size=15),
                },
            )
            if return_fig and not return_mtx:
                return fig
            elif return_mtx and not return_fig:
                return attn_results

def safe_del(a):
    '''
    Deletes a.
    '''
    try:
        exec(f"del {a}")
    except:
        pass
    torch.cuda.empty_cache()

def get_indices_from_sql_lite(fname:str, trial_id:str)-> List:
    '''
    Returns the indices of the trial_id given a SQL file.
    '''            
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(fname)
    df = pd.read_sql_query("SELECT * from trial_params", conn)
    return list(map(int, df[df.trial_id == trial_id].param_value.values))

global last_time
last_time = None
import time

def get_time(s:str):
    '''
    Gets the latest time.
    '''
    global last_time
    if last_time is None:
        last_time = time.time()

    else:
        print(f"Time elapsed - {s} -: {time.time() - last_time}")
        last_time = time.time()
        

def scatter_attention_and_contribution(
        model: Any,
        layer_no: int,
        head_no: int,
        dataset: Any,
        outputs: Tensor,
        return_vals:bool=False,
        return_fig:bool=False,
):
    '''
    Plots a scatter plot for each input with the attention paid to what we are testing on and the amount written in the decided directions.
    Args:
        model(Any): A model.
        layer_no(int): An integer of layer number.
        head_no(int): An integer of head number.
        dataset(Any): A dataset.
        return_vals(bool): A boolean of whether to return values or not. Defaults to False.
        return_fig(bool): A boolean of whether to return figure or not. Defaults to False.
    Returns:

    '''
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    model_unembed = model.unembed.W_U.detach().cpu()
    df = []
    cache = {}
    model.cache_all(cache)
    input_list=["text","audio","image","3d","time-series","video", "sensor"]

    logits = model(outputs)
    prompts = labels

    for i, prompt in enumerate(prompts):
        # TODO: Add processing logic here
        pass 

    viz_df = pd.DataFrame(
        df, columns=[f"Attention Probability on Inputs", f"Dot w Name Embed", "Input Type", ""]
    )








