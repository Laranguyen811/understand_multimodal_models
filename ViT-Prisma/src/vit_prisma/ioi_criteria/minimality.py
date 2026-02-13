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



