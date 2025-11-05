# %%
import re
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from rich import print as rprint
from rich.table import Table, Column
from torch import Tensor
from tqdm.notebook import tqdm
from vit_prisma.prisma_tools.activation_cache import ActivationCache
from vit_prisma.models.base_transformer import HookedTransformer
from vit_prisma import utils
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.layer_norm import LayerNorm
from vit_prisma.models.layers.patch_embedding import PatchEmbedding
from vit_prisma.models.layers.position_embedding import PosEmbedding
from vit_prisma.prisma_tools.hook_point import HookPoint
from transformers import CLIPProcessor, CLIPModel,CLIPConfig, AutoTokenizer

# %%
# Model and Task Setup
model_name = "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K"
clip_model = load_hooked_model(model_name)

