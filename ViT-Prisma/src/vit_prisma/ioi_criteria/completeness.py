# %%
import os
import torch

assert torch.cuda.device_count() == 1
from statistics import mean
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
import warnings
import json
from numpy import sin, cos, pi
from time import ctime
from dataclasses import dataclass
from vit_prisma.utils.ioi_utils import logit_diff
from tqdm import tqdm
import pandas as pd
from vit_prisma.utils.experiment_utils import (
    to_numpy,
    get_corner,
    print_gpu_mem
)
from vit_prisma.models.activation_fns import quick_gelu
from vit_prisma.prisma_tools.hook_point import HookedRootModule, HookPoint
from vit_prisma.utils.experiments import (
    ExperimentMetric,
    AblationConfig,
    Ablation,
    Patching,
    PatchingConfig,
    get_act_hook,
)

from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly 
from sklearn.linear_model import LinearRegression
import random
import spacy
import re
from einops import rearrange
from pprint import pprint
import gc
from datasets import load_dataset
import matplotlib.pyplot as plt
import vit_prisma
from vit_prisma.utils.ioi_utils import (
    basis_change,
    add_arrow,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
    plot_ellipse,
    calculate_probs,
    calculate_logits_to_ave_logit_diff
)

from vit_prisma.utils.ioi_circuit_extraction import (
    join_lists_nested,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,

)
from copy import deepcopy
from vit_prisma.utils.detect_architectures import detect_architecture
import os

plotly_colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
]
from functools import partial

model_name: str = ""

print_gpu_mem("About to load our model.")
model = vit_prisma.load_model(model_name)
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model = model.to(device)

print_gpu_mem("Model loaded.")

# IOI Dataset Initialisation
dataset = None

def get_all_nodes(circuit:Dict, tokens:Dict)-> List:
    '''
    Gets all nodes in the circuit dictionary.
    Arguments:
        circuit (Dict): The circuit dictionary containing various components.
    Returns:
        List: A list containing all nodes in the circuit.
    '''
    nodes = []
    for circuit_class in circuit:
        for head in circuit[circuit_class]:
            nodes.append((head,tokens[head][0]))
    return nodes

# Import model and dataset

mean_dataset = None

model.reset_hooks()
def measure_faithfulness(circuits:Dict, naive: Dict, dataset:Any, model:Any)-> float:
    '''
    Measures the faithfulness of a given circuit on the dataset using the model.
    Arguments:
        circuits (Dict): The circuit dictionary containing various components.
        naive (Dict): The naive circuit dictionary.
        dataset (Any): The dataset to evaluate the circuit on.
        model (Any): The model used for evaluation.
    Returns:
        float: The faithfulness score of the circuit.
    '''
    nodes = get_all_nodes(circuit)
    # Implement faithfulness measurement logic here
    logit_diff_M = calculate_logits_to_ave_logit_diff(model, dataset)
    print (f"Logit diff on mean dataset: {logit_diff_M}")

    for circuit in [circuits.copy(), naive.copy()]:
        all_nodes = get_all_nodes(circuit)
        heads_to_keep = get_heads_circuit(dataset, excluded=[],circuit=circuit)
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            dataset=dataset,
            mean_dataset=mean_dataset
        )
        logit_diff_circuit = calculate_logits_to_ave_logit_diff(model, dataset)
        print (f"Logit diff after circuit extraction: {logit_diff_circuit}")
    
    circuit = deepcopy(naive)
    print("Working with", circuit)
    cur_metric = logit_diff # Define the current metric

    run_original = True
    print("Are we running the original experiment?", run_original)

    if run_original:
        circui_perf = []
        perf_by_sets = []
        for G in tqdm(list(circuit.keys())+["none"]):
            if G == "ablation":
                continue
            print_gpu_mem(G)
            excluded_classes = []
            if G != "none":
                excluded_classes.append(G)
            heads_to_keep = get_heads_circuit(
                dataset, excluded=excluded_classes, circuit=circuit
            )
            model.reset_hooks()
            model, _ = do_circuit_extraction(
                model=model,
                heads_to_keep=heads_to_keep,
                mlps_to_remove={},
                dataset=dataset,
                mean_dataset=mean_dataset
            )
            torch.cuda.empty_cache() # Clear GPU memory
            cur_metric_broken_circuit, std_broken_circuit = cur_metric(
                model, dataset, std=True, all=True
            ) # Measure the current metric broken circuit
            # Adding back the whole model
            excl_class = list(circuit.keys())
            if G != "none":
                excl_class.remove(G) # If we are adding back G, exclude all other classes but G
            G_heads_to_remove = get_heads_circuit(
                dataset, excluded=excl_class, circuit=circuit
            )
            torch.cuda.empty_cache() # Clear GPU memory

            model.reset_hooks()
            model, _ = do_circuit_extraction(
                model=model,
                heads_to_keep=G_heads_to_remove,
                mlps_to_remove={},
                dataset=dataset,
                mean_dataset=mean_dataset
            )

            torch.cuda.empty_cache() # Clear GPU memory
            cur_metric_coble, std_coble_circuit = cur_metric(
                model, dataset, std=True,all=True
            ) # Measure the current metric coble circuit
            print(cur_metric_coble.mean(), cur_metric_broken_circuit.std())
            torch.cuda.empty_cache()

            # Metric (M\G)
            on_diagonals =[] # List to store on-diagonal values
            off_diagonals = [] # List to store off-diagonal values
            for i in range(len(cur_metric_coble)):
                circui_perf.append({
                    "removed_set_id": G,
                    "ldiff_broken":float(cur_metric_broken_circuit[i].cpu().numpy()),
                    "ldiff_coble":float(cur_metric_coble[i].cpu().numpy()),
                    "input":dataset.inputs[i],
                    "template":dataset.templates[i],
                })

                x, y = basis_change(
                    circui_perf[-1]["ldiff_broken"],
                    circui_perf[-1]["ldiff_coble"],
                )
                model_architecture = detect_architecture(model)

                if model_architecture in ['bert', 'gpt2','gpt_neo','gpt_generic','t5','llama','mistral','claude','falcon', 'mpt', 'bloom','opt']:
                    circui_perf[-1]["on_diagonal"] = x
                    circui_perf[-1]["off_diagonal"] = y
                    layer_0_pattern_from_cache = 
                    on_diagonals.append(x)
                    off_diagonals.append(y)
                else:
                    print("Architecture does not have on and off diagonal metrics.")
            perf_by_sets.append({
                "removed_group": G,
                "mean_cur_metric_broken": float(mean([item["ldiff_broken"] for item in circui_perf if item["removed_set_id"] == G])),
                "mean_cur_metric_coble": float(mean([item["ldiff_coble"] for item in circui_perf if item["removed_set_id"] == G])),
                "std_cur_metric_broken": float(std_broken_circuit.cpu().numpy()),
                "std_cur_metric_coble": float(std_coble_circuit.cpu().numpy()),
                "mean_on_diagonal": float(mean(on_diagonals)) if len(on_diagonals) > 0 else None, # Calculate mean on-diagonal if available
                "mean_off_diagonal": float(mean(off_diagonals)) if len(off_diagonals) > 0 else None, # Calculate mean off-diagonal if available
                "std_on_diagonal": float(np.std(on_diagonals)) if len(on_diagonals) > 0 else None,
                "std_off_diagonal": float(np.std(off_diagonals)) if len(off_diagonals) > 0 else None,
                "color": CLASS_COLORS[G],
                "symbol": "diamond-x",
            })
            perf_by_sets[-1]["mean_abs_diff"]= abs(
                perf_by_sets[-1]["mean_cur_metric_broken"]
                - perf_by_sets[-1]["mean_cur_metric_coble"]
            ).mean()

            df_circuit_perf = pd.DataFrame(circui_perf)
            circuit_classes = sorted(perf_by_sets, key=lambda x: x["mean_abs_diff"]) # Sort circuit classes by mean absolute difference
            df_perf_by_sets = pd.DataFrame(perf_by_sets)
    
    return (df_circuit_perf, circuit_classes, df_perf_by_sets)

def create_circuit_figures(circuit_perf:List, circuit_classes:List, df_circuit_perf:pd.DataFrame) -> None:
    '''
    Creates circuit figures based on performance by sets and circuit classes.
    Arguments:
        perf_by_sets (List): List of performance metrics by sets.
        circuit_classes (List): List of circuit classes.
        df_circuit_perf (pd.DataFrame): DataFrame containing circuit performance data.
    Returns:
        None
    '''
    with open(f"sets/perf_by_classes_{ctime()}.json","w") as f:
        json.dump(circuit_perf, f)
    
    fig = go.Figure()

    # Add the grey region and make the dotted line
    minx = -2
    maxx = 6
    eps = 1.0
    xs = np.linspace(minx - 1, maxx + 1, 100)
    ys = xs

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"x=y",
            line=dict(color="grey", dash="dash")
        )
    )

    rd_set_added = False
    for i, perf in enumerate(perf_by_sets):
        fig.add_trace(
            go.Scatter(
                x=[perf["mean_cur_metric_broken"]],
                y=[perf["mean_cur_metric_coble"]],
                mode="markers",
                name=perf["removed_group"],
                marker=dict(
                    color=perf["color"],
                    size=10,
                    symbol=perf["symbol"],
                ),
                    showlegend=(
                        ("1" in perf["removed_group"][-2:])
                        or ("Set" in perf["removed_group"])
                    ),
                )
            )

    fig.update_xaxes(title_text="F(C \ K)")
    fig.update_yaxes(title_text="F(M \ K)")
    fig.update_xaxes(showgrid=True,gridcolor="black",gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="black",gridwidth=1)
    fig.update_layout(paper_bgcolor="white", plot_bgcolors="white")

    # Use these lines to scale Scalable Vector Graphics (SVGs) properly 
    fig.update_xaxes(range=[minx, maxx])
    fig.update_yaxes(range=[minx, maxx])
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

    fig.update_yaxes(scaleanchor="x", scaleratio=1,)

    circuit_to_export = "natural"
    fpath = f"circuit_completeness_{circuit_to_export}_CIRCUIT_at_{ctime()}.svg"
    if os.path.exists("/path/to/svgs"):
        fpath = "svgs" + fpath  
