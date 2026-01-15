import os
import torch

assert torch.cuda.device_count() == 1
from statistics import mean
from IPython.core.getipython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.magics_manager("load_ext autoreload")
    ipython.magics_manager("autoreload 2")
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



