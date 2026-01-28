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
    
    







