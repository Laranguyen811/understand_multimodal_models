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
)





