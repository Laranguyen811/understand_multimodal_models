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
from vit_prisma.utils.data_utils.visual_genome.visual_genome_objects import create_dict, base_dir
from vit_prisma.utils.data_utils.visual_genome.visual_genome_relationships import create_rel_data, rel_base_dir, get_relationships_by_predicate
import torch.nn.functional as F
import pandas as pd
# %%
# Loading the model 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# Verify that the model can do the task
#visual_genome_dataset, visual_genome_loader = instantiate_dataloader()
loaded_images = load_images(train_path=set_up_vg_paths(verbose=True),batch_size=8,verbose=False)
print(f"Number of loaded images: {len(loaded_images)}")
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
text_vision_outputs = processor(text=text,images=loaded_images[0],return_tensors='pt')
rel_data = create_rel_data(rel_base_dir=rel_base_dir)

holding_cases = get_relationships_by_predicate(image_id=1,relationships_data=rel_data, predicate='holding')
print(f"Holding case: {holding_cases}")
# %%

# Build IOI test
def build_ioi_test_case(cases,obj_dict):
    '''
    Build IOI test case from relationship data.
    Args:
        cases: List of relationship cases.
        obj_dict: Dictionary mapping object IDs to object names.
    Returns:
        text: List of text prompts.
        image: Corresponding image.
    
    '''
    image_id = cases[0]['image_id'] 
    correct_object = cases[0]['object']
    all_objects = obj_dict.get(image_id,[])
    distractors = [obj for obj in all_objects if obj != correct_object]
    if distractors is None or len(distractors) == 0:
        return None,None
    
    return {
        'image_id': image_id,
        'subject': cases[0]['subject'],
        'correct_object': correct_object,
        'distractors': distractors[0],
        'texts' : [correct_object,distractors[0]],
        'correct_idx':0,
        'distractor_idx':1
    }
# Example usage
ioi_test_case = build_ioi_test_case(holding_cases,obj_dict)
print(f"IOI Test Case: {ioi_test_case}")
text = ioi_test_case['texts']
text = [f"An image of {obj}" for obj in text]
print(f"IOI Test Text Prompts: {text}")


# %%
ioi_inputs = processor(text=text, images=loaded_images[0],return_tensors='pt',padding=True)
print(f"Input type: {type(inputs)}")

with t.no_grad():
    
    # Image and text Embeddings
    image_embeds = clip_model.get_image_features(pixel_values=ioi_inputs['pixel_values'])
    text_embeds = clip_model.get_text_features(input_ids=ioi_inputs['input_ids'])

    outputs = clip_model(**ioi_inputs)
    # Logits
    logits = outputs.logits_per_image[0]

    # Probabilities
    probs = logits.softmax(dim=-1)
print(f"Logit values: {logits}")
print (f"Probabilities: {probs}")
print(f"Correct IOI probability: {np.round(probs[ioi_test_case['correct_idx']].item(),3)}")
print(f"Distractor IOI probability: {np.round(probs[ioi_test_case['distractor_idx']].item(),3)}")

def calculate_cosine_similarity(
        image_embeds: Tensor,
        text_embeds: Tensor,
        correct_idx: int = 0,
        distractor_idx: int = 1,
        verbose: bool = True,
):
    '''
    Calculate cosine similarity between image and text embeddings.
    Args:
        image_embeds: Image embeddings tensor of shape (batch, d_model).
        text_embeds: Text embeddings tensor of shape (num_texts, d_model).
    Returns:
        Cosine similarity tensor of shape (batch, num_texts).
    '''
    # Normalize the embeddings since magnitudes do not influence cosine similarity
    image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)
    correct_text_embed = text_embeds_norm[correct_idx]
    distractor_text_embed = text_embeds_norm[distractor_idx]
    if verbose:
        print(f"Shape of normalized image embeddings: {image_embeds_norm.shape}")
        print(f"Shape of normalized correct text embedding: {correct_text_embed.shape}")
        print(f"Shape of normalized distractor text embedding: {distractor_text_embed.shape}")
    # Calculate cosine similarity
    cos_sim_correct =(image_embeds_norm @ correct_text_embed).squeeze(0)
    cos_sim_distractor = (image_embeds_norm @ distractor_text_embed).squeeze(0)
    if verbose:
        print(f"Cosine similarity with correct text: {cos_sim_correct.item():.3f}")
        print(f"Cosine similarity with distractor text: {cos_sim_distractor.item():.3f}")
    cos_sim_diff = cos_sim_correct - cos_sim_distractor

    return cos_sim_diff



cos_sim_diff = calculate_cosine_similarity(image_embeds, text_embeds)
print(f"Cosine similarity difference between correct and distractor: {cos_sim_diff.item():.3f}")

def calculate_logits_to_ave_logit_diff(
        logits: Tensor,
        per_prompt: Bool = False,
    ) -> Float[Tensor, "batch"]:
    '''
    Returns the average logit difference between the correct and distractor prompts.
    Args:
        logits: Logits tensor of shape (batch, seq, d_model).
        per_prompt: If True, returns the logit difference per prompt.
    Returns:
        Average logit difference tensor of shape (batch,).
    '''
    # Calculate the logit difference between correct and distractor
    logits_diff = logits[:, ioi_test_case['correct_idx']] - logits[:, ioi_test_case['distractor_idx']]
    # If per_prompt is True, return the logit difference per prompt
    print(f"Logits difference tensor shape: {logits_diff.shape}")
    return np.round(logits_diff.item(),3) if per_prompt else np.round(logits_diff.mean().item(),3)

logits_diff = calculate_logits_to_ave_logit_diff(logits.unsqueeze(0))
print (f"Logits difference between correct and distractor: {logits_diff.item()}")
    
metric_df = pd.DataFrame({
    'Metric': ['Probability Difference':, 'Cosine Similarity Difference', 'Logits Difference'],
    'Value': [np.round(probs[ioi_test_case['correct_idx']].item() - probs[ioi_test_case['distractor_idx']].item(),3), np.round(cos_sim_diff.item(),3), logits_diff]
})
metric_df.style.set_properties(**{'white-space': 'pre-wrap'})
print(metric_df)






