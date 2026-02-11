import os
import torch

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
    ExperimentMetric
)
