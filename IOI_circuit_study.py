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
from vit_prisma.models.model_loader import load_hooked_model
from transformers import CLIPProcessor, CLIPModel,CLIPConfig, AutoTokenizer
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from vit_prisma.utils.prisma_utils import test_prompt
from vit_prisma.dataloaders.visual_genome import set_up_vg_paths, transform, instantiate_dataloader, iterate_dataloader, load_images, load_dataset

# %%
# Loading the model 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# Verify that the model can do the task
#visual_genome_dataset, visual_genome_loader = instantiate_dataloader()
loaded_images = load_images(train_path=set_up_vg_paths(verbose=True),batch_size=8)
#inputs = processor(image=loaded_images,return_tensors='pt', padding = True)
# Get embeddings
#with t.no_grad:
#    image_features = clip_model.get_image_features(**inputs)
#print(f"First image:{(loaded_images)}")
test_prompt(inputs,clip_model)




