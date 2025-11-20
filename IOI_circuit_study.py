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
from vit_prisma.dataloaders.visual_genome import VisualGenomeDataset, set_up_vg_paths, transform, instantiate_dataloader, iterate_dataloader, load_images
from vit_prisma.dataloaders.coco import CocoCaptions
from vit_prisma.utils.data_utils.visual_genome.visual_genome_objects_list import create_dict, base_dir
# %%
# Loading the model 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# Verify that the model can do the task
#visual_genome_dataset, visual_genome_loader = instantiate_dataloader()
loaded_images = load_images(train_path=set_up_vg_paths(verbose=True),batch_size=8)
labels = create_dict(base_dir=base_dir).values()
loaded_images_data = VisualGenomeDataset(loaded_images,labels=labels,transform=transform)
print(f"First image:{(loaded_images)}")
print(f"Type of loaded images:{type(loaded_images)}")
inputs = processor(images=loaded_images,return_tensors='pt',paddings=True)
print(f"Input type: {type(inputs)}")

input_tensor = inputs['pixel_values']

# %%
obj_dict = create_dict(base_dir=base_dir)
first_key,first_values = next(iter(obj_dict.items()))
res = ', '.join(first_values)
res = res.rsplit(', ', 1) # Split at the last comma
res = ' and '.join(res) if len(res) == 2 else res[0]
text = [f"An image of {res}"]
print(text)
text_vision_outputs = processor(text=text,images=loaded_images[0],return_tensors='pt',paddings = True)

with t.no_grad():
    
    # Image and text Embeddings
    image_embeds = clip_model.get_image_features(pixel_values=text_vision_outputs['pixel_values'])
    text_embeds = clip_model.get_text_features(input_ids=text_vision_outputs['input_ids'])

    outputs = clip_model(**text_vision_outputs)
    # Logits
    logits = outputs.logits_per_image[0]








