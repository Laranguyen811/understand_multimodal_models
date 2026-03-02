# %%
import os
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

assert torch.cuda.device_count() == 1
import warnings
from time import ctime
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from vit_prisma.utils.ioi_utils import calculate_probs, calculate_logits_to_ave_logit_diff
from vit_prisma.models.activation_fns import gelu_new
from vit_prisma.utils.experiment_utils import (
    to_numpy,
    get_corner,
    print_gpu_mem,
) # Helper functions

from vit_prisma.prisma_tools.hook_point import (
    HookedRootModule,
    HookPoint
)
from vit_prisma.utils.experiments import (
    ExperimentMetric,
    AblationConfig,
    Ablation,
    Patching,
    PatchingConfig,
    get_act_hook,
)
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
from torch import Tensor
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
import random 
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
from IPython.core import getipython
import matplotlib.pyplot as plt
import random as rd

from vit_prisma.utils.ioi_utils import (
    ALL_COLORS,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)

from vit_prisma.utils.ioi_circuit_extraction import (
    join_lists_nested,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

ipython = getipython()

if ipython is not None:
    ipython.run_line_magic("load_ext autoreload")
    ipython.run_line_magic("autoreload 2")
def get_circuit_extraction(CIRCUIT: Dict, NAIVE: Dict, dataset: Any, mean_dataset:Any)-> None:
    '''
    Specifies a model and performs circuit extraction.
    Args:
        CIRCUIT(Dict): A dictionary of circuits.
        NAIVE(Dict): A dictionary of naive circuits.
    Returns:
        None 
    '''
    model_name: str = " "
    print_gpu_mem("About to load model")
    model = None
    model.set_use_attn_result(True)
    device = "cuda"
    if torch.cuda.is_available():
        model.to(device)

    print_gpu_mem("{model_name} loaded.")
    N = 100
    dataset = None
    mean_dataset = None
    circuits = [None, CIRCUIT.copy(), NAIVE.copy()]
    circuit = circuits[1]

    metric = calculate_logits_to_ave_logit_diff
    naive_heads = []
    for heads in circuits[2].values():
        naive_heads += heads
    model.reset_hooks()
    model_baseline_metric = metric(model, dataset)
    
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep={},
        mlps_to_remove={},
        dataset=dataset,
        mean_dataset=mean_dataset,
        excluded=naive_heads,
    )
    
    circuit_baseline_metric = metric(model, dataset)
    print(f"{model_baseline_metric} {circuit_baseline_metric}")
    return circuit, circuits
def get_basic_extracted_model(
        circuits:Dict, model: Any, dataset: Any, mean_dataset: Any=None,
):
    '''
    Obtains basic extracted model. 
    Args:
        circuits (Dict): A dictionary of circuits.
        model(Any): A model.
        dataset(Any): A dataset.
        mean_dataset(Any): A mean dataset. 
    Returns:

    '''
    circuit = circuits[1]
    if mean_dataset is None:
        mean_dataset = dataset
    heads_to_keep = get_heads_circuit(
        dataset,
        excluded=[],
        circuit=circuit,

    )
    torch.cuda.empty_cache() # Empty cache in cuda

    model.reset_hooks()
    model , _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        dataset=dataset,
        mean_dataset=mean_dataset,
    ) 

    return model

def calculate_metrics(model:Any, dataset: Any, mean_dataset: Any, circuits:Dict, metric:function):
    '''
    Calculates metrics. 
    Args:
        model (Any): A model.
        dataset (Any): A dataset.
    Returns:
        None
    '''
    model = get_basic_extracted_model(
        model,
        dataset,
        mean_dataset,
        circuit=circuits[1]
    )
    torch.cuda.empty_cache()

    circuit_baseline_diff, circuit_baseline_diff_std = calculate_logits_to_ave_logit_diff(
        model, dataset, std=True
    )

    torch.cuda.empty_cache()
    circuit_baseline_prob, circuit_baseline_prob_std = calculate_probs(model, dataset, std=True)
    torch.cuda.empty_cache()
    model.reset_hooks()
    baseline_ldiff, baseline_ldiff_std = calculate_logits_to_ave_logit_diff(model, dataset, std=True)
    torch.cuda.empty_cache()
    baseline_prob, baseline_prob_std = calculate_probs(model, dataset, std=True)

    print(f"{circuit_baseline_diff}, {circuit_baseline_diff_std}")
    print(f"{circuit_baseline_prob}, {circuit_baseline_prob_std}")
    print(f"{baseline_ldiff}. {baseline_ldiff_std}")
    print(f"{baseline_prob}, {baseline_prob_std}")

    if metric == calculate_logits_to_ave_logit_diff:
        circuit_baseline_metric = circuit_baseline_diff
    else:
        circuit_baseline_metric = circuit_baseline_prob
    
    circuit_baseline_metric = [None, circuit_baseline_metric, circuit_baseline_metric]
    return circuit_baseline_metric

def define_set_K(circuit:Dict, circuits:Dict):
    '''
    Defines the sets K for every vertex in the graph.
    Args:
        circuit(Dict): A dictionary of circuit.
        K (Dict): A dictionary of sets K.
    Returns:
        None
    '''
    K = dict()
    for circuit_class in circuit.keys():
        for head in circuit[circuit_class]:
            K[head] = [circuit_class]

    # Rebuild J
    for head in K.keys():
        new_j_entry = []
        for entry in K[head]:
            if isinstance(entry, str):
                for head2 in circuits[1][entry]: 
                    new_j_entry.append(head2)
            elif isinstance(entry, tuple):
                new_j_entry.append(entry)
            else:
                raise NotImplementedError(head,entry)
        assert head in new_j_entry, (head, new_j_entry)
        K[head] = list(set(new_j_entry))

def run_experiment(circuit:Dict, circuits:Dict, K: Dict, dataset: Any,mean_dataset: Any, model: Any, metric:function)-> Dict:
    '''
    Runs the experiment.
    Args:
        circuit: A dictionary of a circuit.
        circuits: A dictionary of circuits. 
        K: A dictionary of sets K. 
    
    Returns:
        None
    '''
    metric = calculate_logits_to_ave_logit_diff
    results = {}
    if "results_cache" not in dir():
        results_cache = {} # Massively speeds up future runs
    
    for circuit_class in circuit.keys():
        for head in circuits[1][circuit_class]:
            results[head] = [None, None]
            base = frozenset(K[head])
            summit_list = deepcopy(K[head])
            summit_list.remove(head)
            summit = frozenset(summit_list)

            for idx, ablated in enumerate([base, summit]):
                if ablated not in results_cache:
                    new_heads_to_keep = get_heads_circuit(
                        dataset, excluded=ablated, circuit=circuit  
                    )

                    model.reset_hooks()
                    model, _ = do_circuit_extraction(
                        model=model, 
                        heads_to_keep=new_heads_to_keep,
                        mlps_to_remove={},
                        dataset=dataset,
                        mean_dataset=mean_dataset,
                    )
                    torch.cuda.empty_cache()
                    metric_calc = metric(model, dataset, std=False)
                    results_cache[ablated] - metric_calc
                    print("Calculate metrics.")
                results[head][idx] = results_cache[ablated]

            print(f"{head} with {K[head]}: progress from {results[head][0]} to {results[head][1]}")
            return results
        
def plot_figure(model:Any,circuit:Dict, head_positions:List[set], results: Dict):
    '''
    Plots the figure of the metrics.
    Args:
        model(Any): A model.
        circuit(Any): A dictionary of circuit.
    Returns:  
    
    '''
    ac = ALL_COLORS
    cc= CLASS_COLORS.copy()

    relevant_classes = list(circuit.keys())
    fig = go.Figure()
    
    initial_y_cache = {}
    final_y_cache = {}

    the_xs = {}

    for j, G in enumerate(relevant_classes + ["compositional linker"]):
        # Loops throught the relevant classes and the head that focuses the position of the query to the position of the correct object, allowing for compositional link in the image domain 
        xs = []
        initial_ys = []
        final_ys = []
        colors = []
        names = []
        widths = []

        if G == "compositional linker":
            curvys = list(circuit["linker"]) # Creates a dictionary curvyes to store linker circuit activations
            for head in head_positions:
                curvys.remove(head)
        
        elif G == "linker":
            curvys = head_positions
        else:
            curvys=list(circuit[G])
        
        curvys = sorted(curvys, key=lambda x: -abs(results)[x][1] - results[x][0])

        for v in curvys:
            colors.append(cc[G])
            xs.append(str(v))
            initial_y = results[v][0]
            final_y = results[v][1]

            initial_ys.append(initial_y)
            final_ys.append(final_y)

        the_xs[G] = xs
        initial_ys = torch.Tensor(initial_ys)
        final_ys = torch.Tensor(final_ys)
        initial_y_cache[G] = initial_ys
        final_y_cache[G] = final_ys

        y = final_ys - initial_ys

        if True:
            base = [0.0 for _ in range(len(xs))]
            warnings.warn("Base is 0.")
            y = abs(y)
        else:
            base = initial_ys

        fig.add_trace(
            go.Bar(
                x=xs,
                y=y,
                base=base,
                marker_color=colors,
                width=[1.0 for _ in range(len(xs))],
                name=G,
            )
        ) 

    fig.update_layout(
        xaxis_title = "Attention head",
        yaxis_title = "Change in logit difference",
    )

    fig.update_xaxes(
        gridcolor = "black",
        gridwidth=0.1,

    )
    fig.update_layout(legend=dict(yanchor="top",y=0.90,xanchor="left",x=0.01))
    fig.update_yaxes(gridcolor="black",gridwidth=0.1)
    




    








