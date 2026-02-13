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
from vit_prisma.prisma_tools import factored_matrix, activation_cache
from vit_prisma.dataloaders.visual_genome import VisualGenomeDataset, set_up_vg_paths, transform, instantiate_dataloader, iterate_dataloader, load_images
from vit_prisma.utils.data_utils.visual_genome.visual_genome_objects import create_dict, base_dir
from vit_prisma.utils.data_utils.visual_genome.visual_genome_relationships import create_rel_data, rel_base_dir, get_relationships_by_predicate
from vit_prisma.prisma_tools.hook_point import HookPoint
import torch.nn.functional as F
from IPython.display import display
from typing import Any
from einops import einsum
from argparse import Namespace
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
import random
from tqdm import tqdm as tqdm
import random
import plotly.express as px
import pandas as pd
from vit_prisma.utils.detect_architectures import detect_architecture

#from vit_prisma.utils.patching_utils import path_patching, direct_path_patching, direct_path_patching_up_to

# %%
# Loading the model 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"Loaded CLIP model: {clip_model}")
print(f"Type of CLIP model:{type(clip_model)}")
# %%
# Study Compositional Copying (IOI) task, copying an image feature given a text cue, setup 
# Load images and labels
loaded_images = load_images(train_path=set_up_vg_paths(verbose=True),batch_size=8,verbose=False)
print(f"Number of loaded images: {len(loaded_images)}")
objs,obj_dict = create_dict(base_dir=base_dir)
labels = obj_dict.values()
loaded_images_data = VisualGenomeDataset(loaded_images,labels=labels,transform=transform)
print(f"First image:{(loaded_images)}")
print(f"Type of loaded images:{type(loaded_images)}")
# Process images
inputs = processor(images=loaded_images,return_tensors='pt',paddings=True)
print(f"Input type: {type(inputs)}")

input_tensor = inputs['pixel_values']

# %%

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
print(f"Type of log probabilities:{type(probs)}")
correct_prob = np.round(probs[ioi_test_case['correct_idx']].item(),3)
distractor_prob = np.round(probs[ioi_test_case['distractor_idx']].item(),3)
print(f"Correct IOI probability: {correct_prob}")
print(f"Distractor IOI probability: {distractor_prob}")
probability_diff = np.round((correct_prob - distractor_prob),3)
print(f"Difference in probabilities: {probability_diff}")
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
    cos_sim_diff = np.round((cos_sim_correct - cos_sim_distractor).item(),3)

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
    # Calculate the logit difference between correct and distractor rounded to 3 decimal places
    logits_diff = np.round((logits[:, ioi_test_case['correct_idx']] - logits[:, ioi_test_case['distractor_idx']]).item(),3)
    # If per_prompt is True, return the logit difference per prompt
    
    # Calculate mean logit difference rounded to 3 decimal places
    mean_logits_diff = np.round(logits_diff.mean().item(),3)
    
    return logits_diff if per_prompt else mean_logits_diff

logits_diff = calculate_logits_to_ave_logit_diff(logits.unsqueeze(0))
print (f"Logits difference between correct and distractor: {logits_diff}")

# %%
# Patch-level IOI analysis

def find_object_bbox(
        objs: dict,
        object_name: str,
    ) -> list | None:
    '''
    Find object in the list of objects.
    Args:
        objects_data: List of object names.
        object_name: Name of the object to find.
    Returns:
        bounding boxes for the object[x, y, w, h] or None if not found
    '''
    for obj in objs:
        if object_name in obj['names']:
            return [obj['x'], obj['y'], obj['w'], obj['h']]
    return None

# Example usage
chin_bbox = find_object_bbox(objs[0]['objects'], 'chin')
print(f"Bounding box for 'chin': {chin_bbox}")

def get_patch_embeddings(
        model: Any,
        inputs,
    ):
    '''
    Get patch embeddings from the model.
    Args:
        model: Any models.
        image_tensor: Image tensor of shape (batch, channels, height, width).
    Returns:
        Patch embeddings tensor of shape (num_patches, d_model).
    '''
    # Obtain patch embeddings
    with t.no_grad():
        # Get vision model outputs with hidden states
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True,
        )
        vision_patch_embeddings = vision_outputs.last_hidden_state[:,1:, :] # (batch, num_patches, hidden_dim), shape [1, 50, 768]. Obtain the last hidden state because it contains the patch embeddings.
        print(f"Patch embeddings shape: {vision_patch_embeddings.shape}")
        print(f"Type of patch embeddings: {type(vision_patch_embeddings)}")
        # Text embeddings
        text_outputs = model.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'] # Need attention mask for proper processing because of padding. Tells model to ignore padding
        )
        text_embeddings = text_outputs.pooler_output # (batch, hidden_dim), shape [2, 512]
        
        # Projected to shared embedding space. Use projected embeddings for same dimensionality
        patch_embeddings_proj = model.visual_projection(vision_patch_embeddings) # (batch, num_patches, projection_dim), shape [1, 50, 512]
        text_embeddings_proj = model.text_projection(text_embeddings) # (batch, projection_dim), shape [2, 512]
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Type of text embeddings: {type(text_embeddings)}")

        # Normalise embeddings
        vision_patch_embeddings_proj_norm = F.normalize(patch_embeddings_proj, p=2, dim=-1)
        text_embeddings_proj_norm = F.normalize(text_embeddings_proj, p=2, dim=-1)
        print(f"Normalized projected patch embeddings shape: {vision_patch_embeddings_proj_norm.shape}")
        print(f"Normalized projected text embeddings shape: {text_embeddings_proj_norm.shape}")

        # Calculate cosine similarities per patch
        cosine_similarities = einsum(vision_patch_embeddings_proj_norm,text_embeddings_proj_norm, "b n d, t d -> n t") # (batch num_patches d_model x num_texts dim -> num_patches, batch)
    return vision_patch_embeddings_proj_norm, text_embeddings_proj, cosine_similarities

def find_patch_to_correct_object(
        cosine_similarities: Tensor,
        text: list,
    ):
    '''
    Find the patch index with the highest cosine similarity to the correct object.
    Args:
        cosine_similarities: Cosine similarities tensor of shape (batch, num_patches, num_texts).
        correct_idx: Index of the correct text prompt.
    Returns:
        Top 5 most similar patches.
    '''
    for text_idx in range(len(text)):
        # Get cosine similarities for the correct object
        patch_sims = cosine_similarities[:, text_idx] # (num_patches,)
        top_patches = patch_sims.topk(5)

        print(f"Text:{text[text_idx]}")
        print(f"Top 5 patches: {top_patches}")
        print(f"Top 5 patch indices: {top_patches.indices.tolist()}")
    return top_patches

def get_region_embeddings(
        patch_embeddings: Tensor,
        bbox: list,
        original_image_size:tuple,
        image_size:tuple = (224,224),
        patch_size:int = 32,
) -> Tensor|None:
    '''
    Get region embeddings from patch embeddings based on bounding box.
    Args:
        patch_embeddings: Patch embeddings tensor of shape (num_patches, d_model).
        bbox: Bounding box [x, y, w, h].
        image_size: Size of the image (height, width).
        patch_size: Size of each patch.
    Returns:
        Region embeddings tensor of shape (num_region_patches, d_model).
    '''
    # Convert bounding box to patch coordinates
    orig_height, orig_width,_ = original_image_size
    print(f"Original height: {orig_height}, Original width: {orig_width}")
    num_patches_per_side = image_size[0] // patch_size
    print(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

    # Scale bbox coordinates to match patch grid
    x_start = int(bbox[0] / orig_width * num_patches_per_side) # left most patch column where the object begins. Calculated by dividing the x coordinate of the bbox by the image width and multiplying by number of patches per side. 
    y_start = int(bbox[1] / orig_height * num_patches_per_side) # top most patch row where the object begins. Calculated by dividing the y coordinate of the bbox by the image height and multiplying by number of patches per side.
    x_end = int((bbox[0] + bbox[2]) / orig_width * num_patches_per_side) # right most patch column where the object ends. Calculated by dividing the (x coordinate + width) of the bbox by the image width and multiplying by number of patches per side.
    y_end = int((bbox[1] + bbox[3]) / orig_height * num_patches_per_side) # bottom most patch row where the object ends. Calculated by dividing the (y coordinate + height) of the bbox by the image height and multiplying by number of patches per side.
    print(f"Before adjustment: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    # Round to integers
    x_start = int(x_start)
    y_start = int(y_start)
    x_end = int(x_end + 0.5) # Round up for end coordinates to include the patch containing the edge of the object
    y_end = int(y_end + 0.5) # Round up for end coordinates to include the patch containing the edge of the object

    # Ensure at least one patch is selected
    if x_start == x_end:
        x_end = x_start + 1
    if y_start == y_end:
        y_end = y_start + 1

    # Clamp to valid range [0, num_patches_per_side]
    x_start = max(0, min(x_start, num_patches_per_side - 1))
    y_start = max(0, min(y_start, num_patches_per_side - 1))
    x_end = max(x_start + 1, min(x_end, num_patches_per_side))
    y_end = max(y_start + 1, min(y_end, num_patches_per_side))

    print(f"After adjustment: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    # Extract patches in region (assuming patches arranged in grid)
    region_patches = []
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            patch_idx = y * num_patches_per_side + x
            if 0 <= patch_idx < patch_embeddings.shape[0]:
                region_patches.append(patch_embeddings[patch_idx])
    print(f"Region patches found: {len(region_patches)}")
    if region_patches:
        region_embeddings = t.stack(region_patches).mean(dim=0) # Average embeddings of region patches
        return F.normalize(region_embeddings, p=2, dim=-1) # Normalize region embeddings
    print(f"No patches found.")
    return None

def compare_region_cosine_similarity(
        patch_embeddings: Tensor,
        text_embeddings: Tensor,
) -> float|None:
    '''
    Compare cosine similarity between region embeddings and text embeddings.
    Args:
        patch_embeddings: Patch embeddings tensor of shape (num_patches, d_model).
        text_embeddings: Text embeddings tensor of shape (num_texts, d_model).
    Returns:
     Difference in cosine similarity between correct and distractor region embeddings.
    '''
    correct_bbox = find_object_bbox(objs[0]['objects'], texts[0])
    print(f"Correct bbox: {correct_bbox}")
    distractor_bbox = find_object_bbox(objs[0]['objects'], texts[1])
    print(f"Distractor bbox: {distractor_bbox}")
    correct_region_embeds = get_region_embeddings(patch_embeddings[0], correct_bbox,original_image_size=loaded_images[0].shape)
    print(f"Correct region embeddings: {correct_region_embeds}")
    distractor_region_embeds = get_region_embeddings(patch_embeddings[0], distractor_bbox,original_image_size=loaded_images[0].shape)
    print(f"Distractor region embeddings: {distractor_region_embeds}")
    if correct_region_embeds is None or distractor_region_embeds is None:
        print("Could not find region embeddings for correct or distractor object.")
        return None 
    query_text_embeddings = text_embeddings[0]  
    correct_cos_sim = (correct_region_embeds @ query_text_embeddings).item()
    distractor_cos_sim = (distractor_region_embeds @ query_text_embeddings).item()
    cos_sim_diff = correct_cos_sim - distractor_cos_sim
    if cos_sim_diff > 0:
        print(f"The model accurately prefers the correct object region with cosine similarity difference: {cos_sim_diff:.3f}")
    else:
        print(f"The model inaccurately prefers the distractor object region with cosine similarity difference: {cos_sim_diff:.3f}")
    return cos_sim_diff

# Example usage
texts = ["chin","wall"]
print(f"Shape of loaded image: {loaded_images[0].shape[0]}")
inputs = processor(text=texts, images=loaded_images[0],return_tensors='pt',padding=True)
patch_embeddings, text_embeddings, cosine_similarities = get_patch_embeddings(clip_model, inputs)
correct_patches = find_patch_to_correct_object(cosine_similarities, texts)
region_cos_sim_diff = np.round(compare_region_cosine_similarity(patch_embeddings,text_embeddings),3)
print(f"Region cosine similarity difference between correct and distractor: {region_cos_sim_diff}")
    


# %%  
ioi_metric_df = pd.DataFrame({
'Metric': ['Log Probability Difference', 'Cosine Similarity Difference', 'Logits Difference', 'Correct Log Probability', 'Distractor Log Probability','Patch-level Cosine Similarity Difference'],
    'Value': [np.round(probs[ioi_test_case['correct_idx']].item() - probs[ioi_test_case['distractor_idx']].item(),3), np.round(cos_sim_diff.item(),3), logits_diff, correct_prob, distractor_prob, region_cos_sim_diff]
})
ioi_metric_df.style.set_properties(**{'white-space': 'pre-wrap'})
display(ioi_metric_df)





