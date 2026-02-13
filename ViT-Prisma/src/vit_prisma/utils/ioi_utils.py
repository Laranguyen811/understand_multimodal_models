from contextlib import suppress
import warnings

from traitlets import Float
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
from vit_prisma.utils.ioi_circuit_extraction import do_circuit_extraction
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

def plot_ellipse(fig: Figure, xs, ys, color="MediumPurple", nstd:int=1, name:str=""):
    '''
    Plots ellipse.

    '''
    mu = np.mean(xs), np.mean(ys)
    sigma = np.cov(xs, ys)
    w, h, t = ellip

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
        df, columns=[f"Attention Probability on Inputs", f"Dot w Embed", "Input Type", ""]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attention Probability on Inputs",
        y=f"Dot w Embed",
        color = "Input Type",
        hover_data=[""],
        color_discrete_sequence=["rgb(114,255,100)","rgb(201,165,247)"],
        title=f"How Strong {layer_no}.{head_no} Writes in the Embed Direction Relative to Attention Probability",
    )
    if return_vals:
        return
    if return_fig:
        return fig
    else:
        fig.show()
     

def handle_all_and_std(returning: Tensor, all: bool, std: bool)-> Tensor:
    '''
    Handles all and standard deviation.
    Args:
        returning (Tensor): A returning tensor. 
        all (bool): A boolean of whether it is all or not.
        std (bool): A boolean of whether it is a standard deviation or not.
    Returns:
        A returning Tensor 
    '''
    
    if all and not std:
        return returning
    if std:
        if all:
            first_bit = (returning).detach().cpu()
        else:
            first_bit = (returning).mean().detach().cpu()
        return first_bit, torch.std(returning).detach().cpu()
    return (returning).mean().detach().cpu()

def get_attention_on_token(
        model: Any, 
        dataset: Any, 
        layer: Union[str,int], 
        head_idx:int, 
        token: Tensor,
        all:bool=False,
        std:bool=False,
        scores:bool=False, 

):
    '''
    Gets the attention on token 'token' from the END position. 
    Args:
        model (Any): A model.
        dataset (Any): A dataset.
        layer (Union[str,int]): A string or integer of the layer number.
        head_idx (int): An integer of head index.
        token (Tensor): A tensor of token. 
        all (bool): A boolean of whether it is all or not. Defaults to False.
        std (bool): A boolean of whether it is a standard deviation or not. Defaults to False.
        scores (bool): A boolean of whether there are scores or nt. Defaults to False.
    Returns:
        A returning Tensor
    '''

    hook_name_raw = "blocks.{}.attn.hook_attn" + ("_scores" if scores else "") # Obtain the raw hook name
    hook_name = hook_name_raw.format(layer) # Obtain the formatted hook name
    cache = {}
    model.cache_some(cache, lambda x:x == hook_name)
    logits = model(dataset.toks.long()).detach()
    atts = cache[hook_name][
        torch.arange(dataset.N),
        head_idx
    ]
    return handle_all_and_std(atts, all, std)

def sort_positions(x: torch.Tensor) -> Tensor:
    '''
    Sorts the order of the elements of x.
    Args:
        x (torch.Tensor): A tensor of input x.
    Returns:
        A Tensor of sorted x
    '''
    return torch.argsort(x, dim=-1)

def rank_posses(
        model: Any,
        dataset: Any,
        prompts: str,
        all: bool=False,
        std:bool=False,
) -> Tensor:
    '''
    Ranks of the IO (Indirect Object) token in all the tokens.
    Args:
        model(Any): A model.
        dataset(Any): A dataset.
        all(bool): A boolean of whether it is all or not. Defaults to False.
        std(bool): A boolean of whether it is standard deviation or not. Defaults to False.
    Returns:
        A returning Tensor
    '''
    prompts = prompts
    logits = model(prompts).detach().cpu()
    warnings.warn("+1ing")
    end_logits = logits[
        torch.arange(len(prompts)), dataset.idx[-1] + 1, :
        ]
    positions = torch.argsort(end_logits, dim=1)
    io_positions = positions[torch.arange(len(prompts)), dataset.io_tokenIDs]

    return handle_all_and_std(io_positions, all, std)

def calculate_probs(
        model: Any,
        dataset: Any,
        all: bool=False,
        std: bool=False,
        type:str="io",
        verbose: bool = False,
)-> Tensor:
    '''
    Calculate the IO probabilities. 
    Args:
        model (Any): A model.
        dataset (Any): A dataset.
        all (bool): A boolean of whether it is all or not. Defaults to False.
        std (bool): A boolean of whether it is standard deviation. Defaults to False.
        type (str): A string of type of cases. Defaults to "io".
        verbose (bool): A boolean of verbose or not. Defaults to False.
    Returns:
        A returning Tensor
    '''

    logits = model(
        dataset.toks.long()
    ).detach()

    end_logits = logits[
        torch
    ]
def get_top_tokens_and_probs(
        model: Any,
        prompt: str,
) -> Tuple[Tensor, Tensor]:
    '''
    Gets the top tokens and probabilities.
    Args:
        model (Any): A model.
        prompt (str): A string of prompts.
    Returns:
        A tuple of Tensor of top tokens and Tensor of top probabilities.
    '''
    logits, tokens = model(prompt, return_type="logits_and_tokens")
    logits = logits.squeeze(0) # Remove the batch dimension
    end_probs = torch.softmax(logits, dim=-1) # Obtain the softmax of logits along the last dimension as probabilities
    return end_probs, tokens

def get_all_subsets(L: List)-> List[List]:
    '''
    Gets all subsets of L.
    Args:
        L (List): A list of L.
    Returns:
        A list of all subsets of L.
    '''
    if len(L) == 0:
        return [[]]
    else:
        rest = get_all_subsets(L[1:]) # Recursively get all subsets of the rest of the list from index 1 onwards
        return rest + [[L[0]] + subset for subset in rest] # Return the rest plus the first element concatenated with each subset in rest

def ellipse_arc(x_center:float=0,
                y_center:float=0,
                ax1:List[float]=[1,0],
                ax2:List[float]=[0,1],
                a: int=1,
                b: int=1,
                N:int=100)-> Tuple[List,List]:
    '''
    Creates an ellipse arc.
    Args:
        x_center (float): A float of x-coordinate center of the ellipse. Defaults to 0.
        y_center (float): A float of y-coordinate center of the ellipse. Defaults to 0.
        ax1 (List[float]): A list of floats of axis 1, an orthogonal vector representation. Defaults to [1,0].
        ax2 (List[float]): A list of floats of axis 2, an orthogonal vector representation. Defaults to [0,1].
        a (int): An integer of a, the ellipse parameter. Defaults to 1.
        b (int): An integer of b, the ellipse parameter. Defaults to 1.
        N (int): An integer of N. Defaults to 100.
    Returns:
        A tuple of lists of x and y coordinates.
    '''
    if abs(np.linalg.norm(ax1) -1) > 1e-06 or abs(np.linalg.norm(ax2) -1) > 1e-06:
        raise ValueError("ax1 and ax2 must be unit vectors.")
    if abs(np.dot(ax1,ax2)) > 1e-06:
        raise ValueError("ax1 and ax2 must be orthogonal.")
    t = np.linspace(0, 2 * pi, N) # Create a linear space from 0 to 2pi with N points

    # Ellipse parameterisation with respect to a system of axes of directions a1, a2
    xs = a * cos(t)
    ys = b * sin(t)

    # Rotation matrix
    R = np.array([ax1, ax2]).T

    # Coordinates of the ellipse points with respect to the system of axes [1,0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center
    y = yp + y_center
    return x, y

def ellipse_wht(sigma:float) -> Tuple[float, float, float]:
    '''
    Computes the width, height and theta of an ellipse from the covariance matrix sigma.
    Args:
        sigma (float): A float of sigma.
    Returns:
        A tuple of width, height, and theta.
    '''
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1] # Sort eigenvalues in descending order
    vals, vecs = vals[order], vecs[:,order] # Reorder eigenvalues and eigenvectors

    theta = np.arctan2(*vecs[:,0][::-1]) # Compute the angle theta
    if theta < 0:
        theta += 2 * pi
    width, height = 2 * np.sqrt(vals) # Compute width and height
    return width, height, theta

def plot_ellipse(fig: Figure, xs, ys, color="MediumPurple", nstd:int=1, name:str=""):
    '''
    Plots the ellipse. 
    '''
    mu = np.mean(xs), np.mean(ys)
    sigma = np.cov(xs, ys)
    w, h, t = ellipse_wht(sigma)
    w *= nstd
    h *= nstd
    print(f"Ellipse params - w: {w}, h: {h}, theta: {t} rad")

    x, y = ellipse_arc(
        x_center=mu[0],
        y_center=mu[1],
        ax1=[cos(t), sin(t)],
        ax2=[-sin(t), cos(t)],
        a=w,
        b=h,
        
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(size=20, color=color),
            name=name,
        )
    )

def get_heads_from_nodes(nodes: Dict, dataset: Any)-> Dict:
    '''
    Gets heads from nodes.
    Args:
        nodes (Dict): A dictionary of nodes.
        dataset (Any): A dataset.
    Returns:
        A dictionary of heads
    '''
    heads_to_keep_tok = {}
    for h,t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)
    
    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], dataset)
    return heads_to_keep

def calculate_logits_to_ave_logit_diff(
        logits: Tensor,
        ioi_test_case: Dict,
        per_prompt: bool = False,
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
    logits_diff = np.round((logits[:, ioi_test_case['correct_idx']] - logits[:, ioi_test_case['distractor_idx']]).item(),3)
    # If per_prompt is True, return the logit difference per prompt
    
    # Calculate mean logit difference rounded to 3 decimal places
    mean_logits_diff = np.round(logits_diff.mean().item(),3)
    
    return logits_diff if per_prompt else mean_logits_diff

def circuit_from_nodes_logit_diff(model: Any, 
                                  dataset: Any,
                                  nodes: Dict,)-> Float[Tensor, "batch"]:
    '''
    Calculates the logit difference from the circuit defined by nodes.
    Args:
        model (Any): A model.
        dataset (Any): A dataset.
        nodes (Dict): A dictionary of nodes.
    Returns:
        A float tensor of logit difference.
    '''
    heads_to_keep = get_heads_from_nodes(nodes, dataset)
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        dataset=dataset,
    )
    return calculate_logits_to_ave_logit_diff(model,dataset,all=False)

def basis_change(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    '''
    Changes the basis (1, 0) and (0,1) to the basis 1/sqrt(2) (1,1) and 1/sqrt(2) (-1,1) 
    Args:
        x(Tensor): A tensor of x.
        y(Tensor): An array of y.
    Returns:
        A tuple of 2 new tensors 
    '''
    return (x+y)/np.sqrt(2), (y-x)/np.sqrt(2)


def add_arrow(fig: Figure, end_point, start_point, color="black"):
    '''
    Adds arrow to the figure.
    '''
    x_start, y_start = start_point
    x_end, y_end = end_point

    fig.add_annotation(
        x=x_start,
        y=y_start,
        ax=x_end,
        ay=y_end,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="", # If you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=color,
    )

def compute_next_tok_dot_prod(
        model: Any,
        input: Tensor,
        l: Union[int,str],
        h: Union[int,str],
        batch_size:int = 1,
        seq_tokenised: bool = False,

) -> List:
    '''
    Computes the dot product of the model's next token logits, with the logits of the next token in the sentences. Suupports batch_size > 1.
    Args:
        model(Any): A model.
        input(Tensor): A tensor of input.
        l(Union[int,str]): A string or integer of layer.
        h(Union[int,str]): A string or integer of head.
        batch_size(int): an integer of batch size. Defaults to 1.
        seq_tokenised(bool): A boolean of whether the sequence is tokenised or not. Defaults to False. 
    '''
    assert len(input) % batch_size == 0
    cache = {}
    model.cache_some(
        cache, lambda x: x in [f"blocks.{l}.attn.hook_result"], device="cuda"
    )
    if seq_tokenised:
        toks = input # If sequence is tokenised, toks will be assigned to input
    else:
        toks = model.tokenizer(input, padding=False).input_ids # Otherwise, tokenise input
    
    prod = []
    model_unembed = (
        model.unembed.W_U.detach().cpu()
    ) 
    for i in tqdm(range(len(input)) // batch_size):
        model.run_with_hooks(
            input[i * batch_size : (i + 1) * batch_size],
            reset_hooks_start=False,
            reset_hooks_end=False,
        )
        n_seq = len(input)
        for s in range(basis_change):
            idx = i * batch_size + s

            attn_result = cache[f"blocks.{l}.attn.hook_result"][
                s, : (len(toks[idx]) - 1), h, :
            ].cpu()
            next_tok = toks[idx][1:]

            next_tok_dir = model_unembed[next_tok]

            prod.append(
                torch.einsum("hd,hd->h", attn_result,next_tok_dir)
                .detach()
                .cpu()
                .numpy()
            )
        return prod
    
def get_gray_scale(val:float,min_val:float,max_val:float)->float:
    '''
    Gets Gray Scale.
    '''
    max_col = 255
    min_col = 232
    max_val = max_val
    min_val = min_val
    val = val
    return int(min_col + ((max_col - min_col) / max_val - min_val) * (val - min_val))
    
def get_opacity(val: float, min_val: float, max_val: float)-> float:
    '''
    Gets opacity.
    '''
    max_val = max_val
    min_val = min_val
    return (val - min_val) / (max_val - min_val)
    
def print_toks_with_color(toks: Tensor, 
                              color, 
                              show_low:bool=False, 
                              show_high:bool=False,
                              show_all:bool=False)-> None:
    '''
    Prints tokens with color.
    '''
    min_v = min(color)
    max_v = max(color)
    for i,t in enumerate(toks):
        c = get_gray_scale(color[i], min_v, max_v)
        text_c = 232 if c > 240 else 255
        show_value = show_all
        if show_low and c < 232 + 5:
            show_value = True
        if show_high and c > 255 - 5:
            show_value = True
            
        if show_value:
            if len(str(np.round(color[i],2)).split(".")) > 1:
                val = (
                    str(np.round(color[i],2)).split(".")[0]
                ) 
                + "."
                + str(np.round(color[i],2)).split(".")[1][:2]# If show value, if the length of string of rounded color at i to 2 decimals greater than 1, 
                    # val will be the first value of the said string plus "." plus the second to third values of the said string
            else:
                val = str(np.round(color[i],2))
                
            print(f"\033[48;5;{c}m\033[38;5;{text_c}m{t}({val})\033[0;0m", end="") # Print the  ANSI escape sequence
        else:
            print(f"\033[48;5;{c}m\033[38;5;{text_c}m{t}\033[0;0m", end="") # Print the  ANSI ( American National Standards Institute) escape sequence

def convert_tok_color_scale_to_html(toks: Tensor, color):
    '''
    Converts the token color scale to html.
    '''
    min_v = min(color)
    max_v = max(color)
    # Display min and max color in header
    html = f'<span style="background-color: rgba({255},{0},{0}, {0})">Min: {min_v:.2f} </span>'
    + " "
    + f'<span style="background-color: rgba({255},{0},{0}, {255})">Max: {max_v:.2f}</span>'
    + "<br><br><br>" # HTML string
    for i,t in enumerate(toks):
        op = get_opacity(color[i], min_v, max_v)

        html += f'<span style="background-color: rgba({255},{0},{0}, {op})">{t}</span>'
    return html

def export_tok_col_to_file(
        folder: str,
        head: Union[str,int],
        layer: Union[str,int],
        tok_col,
        toks: Tensor,
        chunk_name
):
    if not os.path.isdir(folder):
        os.mkdir(folder) # If the path is not the directory of folder, make directory for folder
    
    if not os.path.isdir(os.path.join(folder,f"layer_{layer}_head_{head}")):
        os.mkdir(os.path.join(folder,f"layer_{layer}_head_{head}"))

    filename = f"{folder}/layer_{layer}_head_{head}/layer_{layer}_head_{head}_{chunk_name}.html"
    all_html = ""
    for i in range(len(tok_col)):
        all_html += (
            f"<br><br><br>==============Sequence {i}=============<br><br><br>"
            + convert_tok_color_scale_to_html(toks[i], tok_col[i])
        )
    with open(filename, "w") as f:
        f.write(all_html)


def get_sample_activation(
        model: Any, 
        dataset: Any,
        hook_names: List[str],
        n: int
)-> Dict[Any, torch.Tensor]:
    '''
    Samples data.
    Args:
        model(Any): A model.
        dataset(Any): A dataset to sample from.
        hook_names(List[str]): A list of strings of hook names.
        n(int): An integer representing the number of data points to sample.
    Returns:
        A dictionary of cache 
    '''
    data = np.random.choice(dataset, n) # Create randomly sampled data
    cache = {}
    model.reset_hooks()
    model.cache_some(cache, lambda name: name in hook_names)
    _ = model(data)
    model.reset_hooks()
    return cache

def get_head_param(
        model:Any,
        module: str,
        layer:Union[str,int],
        head:Union[str,int],
)->Tensor:
    if module == "OV":
        W_v = model.blocks[layer].attn.W_V[head] # Matrix W_v
        W_o = model.blocks[layer].attn.W_O[head] # Matrix W_o
        W_ov = torch.einsum("hd,bh->db",W_v,W_o) # Matrix W_ov
        return W_ov 
    if module == "QK":
        W_k = model.blocks[layer].attn.W_K[head] # Matrix W_k 
        W_q = model.blocks[layer].attn.W_Q[head] # Matrix W_q
        W_qk = torch.einsum("hd,hb->db", W_q,W_k) # Matrix W_qk 
        return W_qk
    if module =="Q":
        W_q = model.blocks[layer].attn.W_Q[head] # Matrix W_q
        return W_q
    if module == "K":
        W_k = model.blocks[layer].attn.W_Q[head] # Matrix W_k
        return W_k
    if module == "V":
        W_v = model.blocks[layer].attn.W_V[head] # Matrix W_v
        return W_v
    if module == "O":
        W_o = model.blocks[layer].attn.W_O[head] # Matrix W_o
    raise ValueError(f"Module {module} not supported")


def get_hook_name(model: Any,
                  module: str,
                  layer: int,
                  head: int,
                  )-> str:
    '''
    Gets hook names from the model.
    Args:
        model(Any): A model.
        module(str): A string of a module name.
        layer(int): An integer of a layer number.
        head (int): An integer of a head number.
    Returns:
        A string of a hook name
    '''
    assert layer < model.cfg["n_layers"]
    assert head < model.cfg["n_heads"]
    if module == "OV" or module == "QK":
        return f"blocks.{layer}.hook_resid_pre"
    raise NotImplementedError("Module must be either OV or QK.")

def compute_composition(
        model: Any,
        dataset: List[str],
        n_samples: int, 
        l_1: int,
        h_1:int,
        l_2:int,
        h_2:int,
        module_1: str,
        module_2: str,

) -> Tensor:
    '''
    Computes the composition between two different modules.
    Args:
        model (Any): A model.
        dataset (List[str]): A list of strings of a dataset.
        n_samples (int): An integer of n samples.
        l_1 (int): An integer of the first layer number.
        h_1 (int): An integer of the first head number.
        l_2 (int): An integer of the second layer number.
        h_2 (int): An integer of the second head number.
        module_1 (str): A string of the first module.
        module_2 (str): A string of the second module.
    Returns:
        A Tensor of the difference between the composition scores and the baseline    
    '''
    W_1 = get_head_param(model, module_1, l_1, h_1).detach() # Obtain the param matrix for the first module
    W_2 = get_head_param(model, module_2, l_2, h_2).detach() # Obtain the param matrix for the second module
    W_12 = torch.einsum("d b,b c -> d c", W_2, W_1) # Obtain the summation between W_1 and W_2
    comp_scores = []

    baselines = []
    hook_name_1 = get_hook_name(module_1,l_1,h_1) 
    hook_name_2 = get_hook_name(module_2, l_2, h_2)
    activations = get_sample_activation(
        model, dataset, [hook_name_1, hook_name_2], n_samples
    ) # Get the activations from the first hook name and the second hook name
    
    x_1 = activations[hook_name_1].squeeze().detach()
    x_2 = activations[hook_name_2].squeeze().detach()

    # Calculate the composition scores and append to the comp_scores list
    c_12 = torch.norm(torch.einsum("d e, b s e -> b s d", W_12, x_1), dim=-1) # Obtain the composition score W_12
    c_1 = torch.norm(torch.einsum("d e, b s e -> b s d", W_1, x_1 ), dim=-1) # Obtain the composition score of W_1
    c_2 = torch.norm(torch.einsum("d e , b s e -> b s d", W_2, x_2), dim=-1) # Obtain the composition score of W_2
    comp_score = c_12/(c_1 * c_2 * 768 ** 0.5) # Calculate the composition score of all embeddings
    comp_scores.append(comp_score)

    # Compute baseline
    for _ in range(10):
        W_1b = torch.randn(W_1.shape, device=W_1.device) * W_1.std() # Create a tensor of random numbers from a normal distribution in the shape of W_1 multiplied by the standard deviation of W_1 
        W_2b = torch.randn(W_2.shape, device=W_2.device) * W_2.std() # Create a tensor of random numbers from a normal distribution in the shape of W_2 multiplied by the standard deviation of W_2
        W_12b = torch.einsum("db, bc -> dc", W_2b, W_1b) # Summing over b 
        c_12b = torch.norm(torch.einsum("d e , b s e -> b s d", W_12b, x_1), dim=-1) # Sum between W_12b and x_1 over e along the last dimension and normalise the summation
        c_1b = torch.norm(torch.einsum("d e, b s e -> b s d", W_2b, x_1), dim=-1) # Sum between W_1b and x_1 over e along the last dimension and normalise the summation
        c_2b = torch.norm(torch.einsum("d e, b s e -> b s d", W_2b, x_2), dim=-1) # Sum between W_1b and x_1 over e along the last dimension and normalise the summation
        baseline = c_12b/(c_1b * c_2b * 768 ** 0.5) # Calculate the baseline for all embeddings
        baselines.append(baseline)

    return (
        torch.stack(comp_scores.mean().cpu().numpy()
        - torch.stack(baselines).mean().cpu().numpy())
    )

def compute_composition_OV_QK(
    model: Any,
    dataset: List[str],
    n_samples: int,
    l_1: int, 
    h_1: int,      
    l_2: int,
    h_2: int,
    mode: str,
) -> None:
    '''
    Computes composition scores between OV and QK matrices. 
    Args:
        model(Any): A model.
        dataset(List[str]): A list of strings of a dataset.
        n_samples(int): An integer of n samples.
        l_1(int): An integer of the 1st layer number.
        h_1(int): An integer of the 1st head number.
        l_2(int): An integer of the 2nd layer number.
        h_2(int): An integer of the 2nd head number.
        mode(str): A string of mode.
    Returns:
        None
    '''

    assert mode in ["Q", "K"]
    W_OV = get_head_param(model,"OV", l_1, h_1).detach() # Obtain matrix OV
    W_QK = get_head_param(model, "QK", l_2, h_2).detach() # Obtain matric QK

    if mode == "Q":
        W_12 = torch.einsum("d b , d c", W_QK, W_OV) # Sum over d between W_QK and W_OV
    elif mode == "K":
        W_12 = torch.einsum(" b c , b c -> d c", W_OV, W_QK) # Sum over b between transposed OV and QK

def patch_all(z: Tensor,
              source_act: Tensor,
              hook: Tensor
              ):
    '''
    Patches all.
    Args:
        z(Tensor): A tensor of an input.
        source_act(Tensor): A tensor of the source activations.
        hook (Tensor): A tensor of hook.
    Returns:
        A Tensor of source activations 
    '''
    return source_act


    
        

    









