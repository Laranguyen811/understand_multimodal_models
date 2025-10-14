# %%
import sys

import vit_prisma
from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils import prisma_utils
from vit_prisma.prisma_tools.activation_cache import ActivationCache
#from transformers import CLIPProcessor, CLIPModel
import einops
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word
import numpy as np
import torch as t
from fancy_einsum import einsum
from collections import defaultdict
from vit_prisma.visualization.attention_visualisation_robust import plot_javascript, prepare_image
import plotly.graph_objs as go
import plotly.express as px

import matplotlib.colors as mcolors

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from IPython.display import display, HTML
from typing import Callable, List, Union, Optional
from jaxtyping import Float, Int
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import plotly.express as px
import plotly.io as pio
import einops
import random
import umap.umap_ as umap
import matplotlib.pyplot as plt
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from torch.utils.data import DataLoader
import jax
import torch.nn.functional as f
from matplotlib.colors import ListedColormap
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# Set the default renderer for Plotly
pio.renderers.default = "browser"  # or "vscode" or "notebook_connected"
import seaborn as sns
import os
import tqdm
import stat
from transformers import CLIPProcessor, CLIPModel,CLIPConfig, AutoTokenizer
from torchvision.transforms.functional import to_pil_image

# %%
# Set the paths to COCO images and annotation files
#root = os.getenv('DATA_ROOT', '/train2017/train2017')
#annFile = os.getenv('ANN_FILE',"/annotations_trainval2017/annotations/captions_train2017.json")
print("Working Directory:", os.getcwd())
base_dir = os.getenv("directory","")
print(f"Base directory: {base_dir}")
train_path = os.path.join(base_dir, "train2017","train2017")
ann_path = os.path.join(base_dir,"annotations_trainval2017","annotations","captions_train2017.json")
ann_path = os.path.normpath(ann_path)  # Normalize the path to ensure it is correct
print("Full annotation file path:", ann_path)

# Check if train_path exists and is a directory
print("Checking train path...")
print("Exists:", os.path.exists(train_path))
print("Is directory:", os.path.isdir(train_path))
# Check read permissions
print("Readable:", os.access(train_path, os.R_OK))


# Check if ann_path exists and is a file
print("Checking annotation file...")
print("Exists:", os.path.exists(ann_path))
print("Is file:", os.path.isfile(ann_path))

# Check read permissions
print("Readable:", os.access(ann_path, os.R_OK))



# Check file mode and permissions
try:
    file_stat = os.stat(ann_path)
    print("File mode:", oct(file_stat.st_mode))
    print("Owner can read:", bool(file_stat.st_mode & stat.S_IRUSR))
except Exception as e:
    print("Error accessing file stats:", e)


try:
    train_stat = os.stat(train_path)
    print("Train path mode:", oct(train_stat.st_mode))
    print("Owner can read:", bool(train_stat.st_mode & stat.S_IRUSR))
except Exception as e:
    print("Error accessing train path stats:", e)





# %%
# Helper function (ignore)

def plot_image(image):
  plt.figure()
  plt.axis('off')
  plt.imshow(image.permute(1,2,0))

class ConvertTo3Channels:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

transform = transforms.Compose([
    ConvertTo3Channels(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def plot_logit_boxplot(average_logits, labels):
  hovertexts = np.array([[IMAGENET_DICT[i] for _ in range(25)] for i in range(1000)])

  fig = go.Figure()
  data = []

  # if tensor, turn to numpy
  if isinstance(average_logits, torch.Tensor):
      average_logits = average_logits.detach().cpu().numpy()

  for i in range(average_logits.shape[1]):  # For each layer
      layer_logits = average_logits[:, i]
      hovertext = hovertexts[:, i]
      box = fig.add_trace(go.Box(
          y=layer_logits,
          name=f'{layer_labels[i]}',
          text=hovertext,
          hoverinfo='y+text',
          boxpoints='suspectedoutliers'
      ))
      data.append(box)


  means = np.mean(average_logits, axis=0)
  fig.add_trace(go.Scatter(
      x = layer_labels,
      y=means,
      mode='markers',
      name='Mean',
      # line=dict(color='gray'),
      marker=dict(size=4, color='red'),
  ))


  fig.update_layout(
      title='Raw Logit Values Per Layer (each dot is 1 ImageNet Class)',
      xaxis=dict(title='Layer'),
      yaxis=dict(title='Logit Values'),
      showlegend=False
  )

  fig.show()


# %%

def plot_patched_component(patched_head, title=''):
  """
  Use for plotting Activation Patching.
  """

  fig = go.Figure(data=go.Heatmap(
      z=patched_head.detach().numpy(),
      colorscale='RdBu',  # You can choose any colorscale
      colorbar=dict(title='Value'),  # Customize the color bar
      hoverongaps=False
  ))
  fig.update_layout(
      title=title,
      xaxis_title='Attention Head',
      yaxis_title='Patch Number',
  )

  return fig

def imshow(tensor, **kwargs):
    """
    Use for Activation Patching.
    """
    px.imshow(
          prisma_utils.to_numpy(tensor),
          color_continuous_midpoint=0.0,
          color_continuous_scale="RdBu",
          **kwargs,
      ).show()

# %%
# We'll use a text-image transformer. Loading the model

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(model)

# %%
# Collate function

def coco_collate_fn(batch):
    images,captions = zip(*batch)  # Unzip the batch into images and captions
    images = t.stack(images, dim=0)  # Stack images into a single tensor
    return images, list(captions)  # Return images and captions as a list

# Define image transformations
transform = transforms.Compose([
    ConvertTo3Channels(),  # Ensure images are in RGB format
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load the COCO dataset
coco_dataset = CocoCaptions(
    root = train_path,
    annFile = ann_path,
    transform = transform,
)

# Create a DataLoader for the dataset
coco_loader = DataLoader(
    coco_dataset, batch_size=8, shuffle=True, # a batch size of 8 is large enough for vector operations and small enough to fit most GPUs/CPUs
    collate_fn=coco_collate_fn,  # Use the custom collate function
)
# Example: get a batch of images and captions
images, captions = next(iter(coco_loader))  # Return the next iteration of the DataLoader
print("Batch of images:", images)
print("Batch of captions:", captions)
print(f"Shape of images tensor: {images.shape}")  # Should be [batch_size, 3, 224, 224]
print(f"Shape of captions: {len(captions)}")  # Should be the number of captions in the batch


# Get all images and captions from the dataset
#images = []
#captions = []
#for img, cap in zip(coco_loader.dataset, coco_loader.dataset.captions):
#    images.append(img)
#    captions.append(cap)  # Assuming cap is a list of captions, we take the first one
#images = t.stack(images)  # Stack the images into a single tensor
#print(f"Number of total images:{len(images)}\n Shape of images tensor: {images.shape}")


# %%
#from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word
#from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
#from vit_prisma.utils.prisma_utils import test_prompt


# %%
# Loading the image

device = "cuda" if t.cuda.is_available() else "cpu"
model = model.to(device) # type: ignore

# Use the batch of images and labels generated earlier
# images: tensor of shape [batch_size, 3, 224, 224] 

# Prepare text and image inputs for the whole batch

# Comparing corresponding captions with the images
first_captions = [cap_list[0] for cap_list in captions]  # Take the first caption for each image

def trimmed_captions(caption, tokenizer_name="openai/clip-vit-base-patch32", max_token=77):
    """
    Trim the caption to fit the model's max token length.
    Args:
        caption: The caption to be trimmed.
        tokenizer_name: The name of the tokenizer to use.
        max_token: The maximum number of tokens allowed.
    Returns:
        The trimmed caption.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer(caption, return_tensors="pt", truncation=True, max_length=max_token)
    return tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
# Trim the captions to fit the model's max token length
trimmed = [trimmed_captions(c,tokenizer_name ="openai/clip-vit-base-patch32", max_token=77) for c in first_captions]  # Trim captions to fit the model's max token length]

text_inputs = processor(text=trimmed, images=None, return_tensors="pt", padding=True).to(device) 
image_inputs = processor(images=images, return_tensors="pt")
image_inputs["pixel_values"] = image_inputs["pixel_values"].to(device)

with t.no_grad():
    outputs = model(**image_inputs, **text_inputs, output_hidden_states=True)
    image_logits = outputs.logits_per_image  # not per-token logits but computed after the model processes the entire image or text input and prokects them into a shared embedding space
    text_logits = outputs.logits_per_text # not per-token logits but computed after the model processes the entire image or text input and prokects them into a shared embedding space
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

print("Image Logits:", image_logits)
print("Text Logits:", text_logits)
print("Label probs:", probs)
print(f"Text input IDs: {text_inputs.input_ids}")
print(f"Image logits shape: {image_logits.shape}")
print(f"Text logits shape: {text_logits.shape}")

fig = go.Figure(data=[
    go.Bar(name='Image Logits', x=first_captions, y=image_logits.cpu().numpy().flatten()),
    go.Bar(name='Text Logits', x=first_captions, y=text_logits.cpu().numpy().flatten())
])
fig.update_layout(
    barmode='group',
    title='Image and Text Logits',
    xaxis_title='Labels',
    yaxis_title='Logits'
)
fig.show()

# %%
# Caching all activations

from vit_prisma.prisma_tools import activation_cache
from vit_prisma.models.model_loader import load_hooked_model

model_name = "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K"
clip_model = load_hooked_model(model_name)
clip_model.to('cuda') # Move to cuda if available
#clip_images = einops.rearrange(images, 'b c h w -> b h w c')
clip_image_logits, clip_image_cache = clip_model.run_with_cache(
    input = image_inputs["pixel_values"],
    return_cache_object=True,
    remove_batch_dim=True,  # Remove batch dimension for simplicity
)
print(type(clip_image_logits), type(clip_image_cache))
print(f"CLIP Image cache:",clip_image_cache)
# %%

# Accessing attention patterns from the first layer
attn_patterns_from_shorthand = clip_image_cache["pattern",0]
attn_patterns_from_full_name = clip_image_cache["blocks.0.attn.hook_pattern"]
print(f"Attention patterns from shorthand: {attn_patterns_from_shorthand.shape}")
print(f"Attention patterns from full name: {attn_patterns_from_full_name.shape}")
act_names = prisma_utils.get_act_name("pattern", 0)  # Get the names of the activations
print(f"Activation names: {act_names}")
# %%
# Verifying that hook_q, hook_k and hook_pattern are related to each other 

layer_0_pattern_from_cache = clip_image_cache["pattern",0]
hook_q = clip_image_cache["q",0]  # Accessing the hook_q for the first layer
hook_k = clip_image_cache["k",0]
# Check the shapes of the hooks

print(f"Shape of hook_q: {hook_q.shape}")

print(f"Shape of hook_k: {hook_k.shape}")
s,n,d = hook_q.shape
device = hook_q.device
attn_scores = einops.einsum(hook_q, hook_k, 's n h, t n h -> n s t')
# Note: Since CLIP or any other ViT-based models use bidirectional self-attention, we do not apply causal masking (autoaggressive masking) because the model can see the entire sequence, not just the one that comes before.
attn_scores = attn_scores/(d ** 0.5)  # Scale the attention scores
layer_0_pattern_from_q_k = attn_scores.softmax(-1)  # Softmax over the last dimension to get the attention pattern
print(f"Shape of layer 0 pattern from cache: {layer_0_pattern_from_cache.shape}")
print(f"Shape of attention scores from q and k: {layer_0_pattern_from_q_k.shape}")
t.testing.assert_close(layer_0_pattern_from_cache, layer_0_pattern_from_q_k)  # Check if they are numerically close to each other
print("The attention patterns from cache and from q and k are numerically close to each other.")
# %%
def visualise_attention(
       heads: Union[int,List[int]],
        cache: ActivationCache,
        cfg: clip_model.cfg,
        title: str,
        max_width: int,
        batch_idx: int =0,
        attention_type = "hook_pattern",

) -> t.Tensor:
    '''
    Visualise attention patterns for a given set of heads.
    Args:
        heads: List of attention head indices or a single index.
        cache (ActivationCache): Stores intermediate activations from hooked layers, indexed by hook name (string). For attention patterns, use keys like  to access full attention maps.
        cfg (HookedViTConfig): Model config, used to compute layer/head indices.
        title (str): Title for the visualisation (unused here).
        max_width (int): Max width for visualisation (unused here).
        batch_idx (int): Batch index to extract from attention cache.
        attention_type (str): Type of attention to extract (default: "attn_scores")
    Returns:
        t.Tensor: A tensor of staced attention patterns for the heads [num_heads, dest, src]
    '''
    if isinstance(heads,int):
        heads = [heads]
    
    # Create the plotting data
    labels: List[str] = []
    patterns: List[t.Tensor] = []
    for head in heads:
        # Set the label
        layer = head // cfg.n_heads # Number of layer is equal to the floor division between head and the number of heads in cfg
        head_idx = head % cfg.n_heads # Number of head index is equal to the remainder of the division between head and the number of heads in cfg
        labels.append(f"Layer {layer} Head {head_idx}")
        # Get the attention patterns for the head
        hook_name = f"blocks.{layer}.attn.hook_pattern"
        pattern = cache[hook_name][head_idx,:,:]
        patterns.append(pattern)
 
        # Combine the patterns into a single tensor
    return t.stack(patterns, dim=0)  # [heads, dest, src]
    

def get_attn_across_datapoints(attn_head_idx, attn_type="attn_scores"):
    '''
    Obtain attention patterns across all datapoints for a given attention head.
    Args:
        list_of_attn_heads: List of attention head indices.
        all_patterns: List of attention patterns for the heads.
    Returns:
        all_patterns: A tensor of attention patterns for the heads across all datapoints.
    '''
    list_of_attn_heads = [attn_head_idx]

    all_patterns = []
    for batch_idx in range(images.shape[0]):
        patterns = visualise_attention(
            list_of_attn_heads, clip_image_cache, clip_model.cfg, "Attention Patterns", 700, batch_idx, attention_type=attn_type),
        all_patterns.extend(patterns) # Adding to the end of the list
    all_patterns = t.stack(all_patterns, dim=0)  # [batch, heads, dest, src]
    return all_patterns

# Set global seeds
def set_global_seeds(seed_value: int):
    '''
    Set global seeds for reproducibility.
    Args:
        seed_value: The seed value to set.
    Returns:
        None
    '''
    t.manual_seed(seed_value)
    t.cuda.manual_seed(seed_value)
    t.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value) # For NumPy operations
    random.seed(seed_value)  # For Python's random module
    t.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility
    t.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CuDNN

seed_value = 42  # Set your desired seed value
set_global_seeds(seed_value)
# Define a worker init function that sets the seed
def worker_init_fn(worker_id):
    '''
    Worker init function for DataLoader.
    Args:
        worker_id: The ID of the worker.
    Returns:
        None
    '''
    worker_seed = t.initial_seed() % 2**32 # Ensuring that the worker seed is a valid 32-bit integer (within the range of a 32-bit unsigned integer (0 to 2^(32-1)))
    np.random.seed(worker_seed)  # Set the seed for NumPy operations in the worker
    random.seed(worker_seed)  # Set the seed for Python's random module in the worker


# %%
# Corner head: a special localised attention head that often focuses on the corners of the feature map or the input image

#image_logits, image_cache = clip_model.run_with_cache(
#    input = image_inputs["pixel_values"],
#    return_cache_object=True,
#    remove_batch_dim=False,
#    names_filter=lambda name: 'hook_pattern' in name  # Add a filter to capture the full attention matrix

#)
#attn_head_idx = 1  # Index of the attention head to visualize
#batch_idx = 0  # Index of the batch to visualize
#full_patterns = image_cache['blocks.0.attn.hook_pattern']
#corner_pattern = full_patterns[batch_idx, attn_head_idx, :, :]
#print(f"Shape of full patterns: {full_patterns.shape}")  # Should be [batch, query tokens, key tokens]

#print(f"Shape of corner pattern: {corner_pattern.shape}")  # Should be [query tokens, key tokens]
#if hasattr(corner_pattern,'cpu'):
#    patterns_np:np.ndarray = corner_pattern.cpu().numpy()
#elif hasattr(corner_pattern,'numpy'):
#    patterns_np:np.ndarray = corner_pattern.numpy()
#else:
#    patterns_np:np.ndarray = np.array(corner_pattern)
#print(f"Shape of corner pattern: {corner_pattern.shape}")
#print("Attention shape",patterns_np.shape)
#print("Attention range:", patterns_np.min(),"to",patterns_np.max())
#print(f"Shape of images", images.shape)
#print(images)
#image = images[batch_idx].cpu().numpy()
#print(f"Shape of image: {image.shape}")  # Should be [3, 224, 224])
#html_code = plot_javascript([patterns_np], image,cfg=clip_model.cfg, ATTN_SCALING=16, cls_token=False)
#display(HTML(html_code))  # Display the HTML code in the notebook

# %%

# Corner head: a special localised attention head that often focuses on the corners of the feature map or the input image
# Assuming clip_model and images are defined elsewhere in your script
# and are correctly loaded.

# Assuming clip_model and images are defined elsewhere in your script
# and are correctly loaded.
from IPython.display import display, HTML
try:
    t.cuda.empty_cache()
except RuntimeError as e:
    print(f"Error {e}")
image_logits, image_cache = clip_model.run_with_cache(
    input = image_inputs["pixel_values"],
    return_cache_object=True,
    remove_batch_dim=True,
    names_filter=lambda name: 'hook_pattern' in name
)

attn_head_idx = 1
batch_idx = 0
cfg = clip_model.cfg  # Use the actual model config, not HookedViTConfig()
print(f"Config Debug")
print(f"CLIP model config patch size: {cfg.patch_size}")
print(f"CLIP model config image size: {cfg.image_size}")
print(f" Expected patches per side: {cfg.image_size // cfg.patch_size}")
print(f"Expected total patches: {(cfg.image_size // cfg.patch_size) ** 2}")  
print(f"Expected total tokens (with CLS): {(cfg.image_size // cfg.patch_size) ** 2 + 1}")
# Get the full attention patterns [batch, heads, query_tokens, key_tokens]
full_patterns = visualise_attention(attn_head_idx, image_cache,cfg, "Attention Scores", 700,attention_type="hook_pattern")
full_patterns = full_patterns.squeeze(0)
print(f"Full patterns shape: {full_patterns.shape}")

corner_patterns = full_patterns[1:, 1:]
patterns_np: np.ndarray = corner_patterns.cpu().numpy()
image_np: np.ndarray = images[batch_idx].cpu().numpy()
print(f"Corner pattern shape after extraction: {patterns_np.shape}")
html_code = plot_javascript(
        [patterns_np],  # List containing the 2D attention matrix
        [image_np],
        cfg=cfg,  # Use the actual model config
        list_of_names=[f"Corner Head: Attention Head {attn_head_idx}"],
        ATTN_SCALING=2,
        cls_token=True  # This should match your 197x197 matrix
    )
# Save the HTML file
#output_path = 'attention_visualization.html'
#try:
#    with open(output_path, 'w', encoding='utf-8') as f:
#        f.write(html_code)
    
#    abs_path = os.path.abspath(output_path)
#    print(f"\n✅ HTML VISUALIZATION SAVED!")
#    print(f"📁 File: {abs_path}")
#    print(f"🌐 URL: file://{abs_path}")
#    print(f"\nTo view:")
#    print(f"1. Copy this path: {abs_path}")
#    print(f"2. Paste into your browser address bar")
#    print(f"3. Or double-click the file in file explorer")
    
#except Exception as e:
#    print(f"❌ Error saving HTML file: {e}")

# Optional: If you're in Jupyter notebook, you can also display inline
#try:
#    from IPython.display import HTML
#    print(f"\n📖 Displaying in notebook...")
#    HTML(html_code)  # This will show the visualization directly in Jupyter
#except ImportError:
#    print(f"\n💡 Not in Jupyter - open the saved HTML file to view visualization")

# Debug: save HTML to file to check JavaScript errors
#with open('debug_viz.html', 'w') as file:
#    file.write(html_code)
#print("HTML saved to debug_viz.html - open in browser to check for JavaScript errors")

output_path = "attention_visualisation.html"
try: 
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_code)
    abs_path = os.path.abspath(output_path)
    print(f"\n✅ HTML Visualisation Saved.")
    print(f"📁 File: {abs_path}")
    print(f"\nTo view:")
    print(f"1. Find the file in your directory.")
    print(f"2. Double-click to open in your web browser.")
    print(f"3. Or copy this path and paste to your browser address bar: file://{abs_path}")
except Exception as e:
    print(f"❌ Error saving HTML file: {e}")
# Extract the specific head's attention matrix [query_tokens, key_tokens]
# FIXED: Select both batch and head dimensions properly
#if full_patterns.ndim == 4:  # [batch, heads, query, key]
#    corner_pattern = full_patterns[batch_idx, attn_head_idx, :, :]
#elif full_patterns.ndim == 3:  # [heads, query, key] - batch already removed
#    corner_pattern = full_patterns[attn_head_idx, :, :]
#elif full_patterns.ndim == 2:  # [query, key] - already the right shape
#    corner_pattern = full_patterns
#else:
#    raise ValueError(f"Unexpected attention pattern shape: {full_patterns.shape}")

# %%
from vit_prisma.visualization.visualize_attention_js import plot_javascript
from IPython.display import display
attn_head_idx = 1
list_of_attn_heads = attn_head_idx
all_patterns = get_attn_across_datapoints(attn_head_idx=attn_head_idx)
all_patterns = all_patterns[0].squeeze(0).cpu().numpy()
print(f"All patterns shape:{all_patterns.shape}")
list_of_images = images[0]
print(f"Length of list of images: {len(list_of_images)}")
html_code_all = plot_javascript(
    [all_patterns],
    list_of_images=[list_of_images],
    list_of_names=[f"Corner Head All Patterns: Attention Head {attn_head_idx}"],
    ATTN_SCALING=8)
display(HTML(html_code_all))


# %%
# Corner token indices
try:
    t.cuda.empty_cache()
except RuntimeError as e:
    print(f"Error {e}")
full_image_logits, full_image_cache = clip_model.run_with_cache(
    input = image_inputs["pixel_values"],
    return_cache_object=True,
    remove_batch_dim=False,
    names_filter=lambda name: 'hook_pattern' in name
)
total_full_patterns = visualise_attention(attn_head_idx, image_cache,cfg, "Attention Scores", 700,attention_type="hook_pattern")
print(f"Full patterns shape: {total_full_patterns.shape}")
num_patch_side = 14
top_left = 1 # First patch (skip CLASS at 0)
top_right = num_patch_side # Last oatch in the first row
bottom_left = (num_patch_side - 1) * num_patch_side + 1 # First patch in the last row
bottom_right = num_patch_side * num_patch_side # Last patch in the last row

print(f"Corner token indices:")
print(f" Top-left: {top_left}")
print(f"Top-right: {top_right}")
print(f"Bottom-left:{bottom_left}")
print(f"Bottom-right:{bottom_right}")
corner_tokens = [top_left,top_right,bottom_left,bottom_right]
full_corner_patterns = full_patterns[corner_tokens,:]
print(f"Corner Patterns' shape: {corner_patterns.shape}")
if hasattr(full_corner_patterns,'cpu'):
    full_patterns_np = full_corner_patterns.cpu().numpy()
else: full_patterns_np = np.array(full_corner_patterns)
image_np: np.ndarray = images[batch_idx].cpu().numpy()
print(f"Corner pattern shape after extraction: {full_patterns_np.shape}")

patterns_list = []
for i in range(4):
    # Reshape to [1,197] then expand to [197, 197] by repeating
    # This shpws the attention pattern as a row we can visualise
    patterns_2d = patterns_np[i:i+1, :]
    # Repeat the pattern to make it visible in the attention matrix
    patterns_2d = einops.repeat(patterns_2d, 'r c -> r c repeat', repeat = 197)
    patterns_list.append(patterns_2d)
# Verify we have the expected shape for CLIP ViT
#expected_tokens = 197  # 196 patches + 1 CLS for CLIP ViT-B/32
#if corner_pattern.shape[0] != expected_tokens or corner_pattern.shape[1] != expected_tokens:
#    print(f"Warning: Expected {expected_tokens}x{expected_tokens} attention matrix, got {corner_pattern.shape}")

# Convert to numpy
#if hasattr(corner_pattern, 'cpu'):
#    patterns_np: np.ndarray = corner_pattern.cpu().numpy()
#elif hasattr(corner_pattern, 'numpy'):
#    patterns_np: np.ndarray = corner_pattern.numpy()
#else:
#    patterns_np: np.ndarray = np.array(corner_pattern)

print(f"Numpy array shape: {corner_patterns.shape}")
print(f"Attention value range: {corner_patterns.min():.4f} to {corner_patterns.max():.4f}")
print(f"Attention first value:{corner_patterns[0]}")
print(f"Minimum attention data: {np.min(corner_patterns)}")
print(f"Maximum attention data:{np.max(corner_patterns)}")
# Generate HTML visualization
html_code = plot_javascript(
        patterns_list,  # List containing the 2D attention matrix
        [image_np] * 4,
        cfg=cfg,  # Use the actual model config
        list_of_names=[
            f"Top-left Corner (Token {top_left})",
            f"Top-right Corner (Token {top_right})",
            f"Bottom-left Corner (Token {bottom_left})",
            f"Bottom-right Corner (Token{bottom_right})"
        ],
        ATTN_SCALING=2,
        cls_token=True  # This should match your 197x197 matrix
    )
# Save the HTML file
#output_path = 'attention_visualization.html'
#try:
#    with open(output_path, 'w', encoding='utf-8') as f:
#        f.write(html_code)
    
#    abs_path = os.path.abspath(output_path)
#    print(f"\n✅ HTML VISUALIZATION SAVED!")
#    print(f"📁 File: {abs_path}")
#    print(f"🌐 URL: file://{abs_path}")
#    print(f"\nTo view:")
#    print(f"1. Copy this path: {abs_path}")
#    print(f"2. Paste into your browser address bar")
#    print(f"3. Or double-click the file in file explorer")
    
#except Exception as e:
#    print(f"❌ Error saving HTML file: {e}")

# Optional: If you're in Jupyter notebook, you can also display inline
#try:
#    from IPython.display import HTML
#    print(f"\n📖 Displaying in notebook...")
#    HTML(html_code)  # This will show the visualization directly in Jupyter
#except ImportError:
#    print(f"\n💡 Not in Jupyter - open the saved HTML file to view visualization")

# Debug: save HTML to file to check JavaScript errors
#with open('debug_viz.html', 'w') as file:
#    file.write(html_code)
#print("HTML saved to debug_viz.html - open in browser to check for JavaScript errors")

# %%
output_path = "attention_visualisation.html"
try: 
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_code)
    abs_path = os.path.abspath(output_path)
    print(f"\n✅ HTML Visualisation Saved.")
    print(f"📁 File: {abs_path}")
    print(f"\nTo view:")
    print(f"1. Find the file in your directory.")
    print(f"2. Double-click to open in your web browser.")
    print(f"3. Or copy this path and paste to your browser address bar: file://{abs_path}")
except Exception as e:
    print(f"❌ Error saving HTML file: {e}")

# %%
os.environ['CUDA_LAUNCH_BLOCKING'] ="1"



# %%
# Get the embeddings
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds
print(f"Image Embeddings: {image_embeds.shape}")
print(f"Text Embeddings: {text_embeds.shape}")

# %%
# Visualise the embeddings in 2-D space using UMAP
text_labels =[f"Caption {i}" for i in range(len(text_embeds))]  # Create unique labels for text embeddings
image_labels = [f"Image {i}" for i in range(len(image_embeds))]  # Create unique labels for image embeddings
cmap = ListedColormap(['steelblue','salmon'])
def plot_embeddings(image_embeds, text_embeds, labels, modality_labels=None, title="Embedding Space"):
    '''
    Plot the embeddings in 2-D space using UMAP.
    Args:
        image_embeds: Image embeddings tensor of shape [batch_size, d_model].
        text_embeds: Text embeddings tensor of shape [batch_size, d_model].
        labels: List of labels for the embeddings.
        modality_labels: List of modality labels for the embeddings (optional).
        title: Title of the plot.
    Returns:
        None
    '''
    assert len(labels) == len(image_embeds) + len(text_embeds), "Labels length must match the number of embeddings"
    combined_embeds = np.vstack([image_embeds,text_embeds])  # Combine image and text embeddings by stacking them vertically
    print(f"Combined Embeddings Shape: {combined_embeds.shape}")
    reducer = umap.UMAP(n_components=2, n_neighbors=50, random_state=0, min_dist=0.1, metric='cosine') # A dimensionality reduction tool, taking higher-dimensional data and projecting it into a lower-dimensional space
    reduced = reducer.fit_transform(combined_embeds) # Used to plot data points on a 2D scatter plot
    assert reduced.shape[1] == 2, f"UMAP failed: expected 2D output, got {reduced.shape[1]}D"
    if reduced.ndim != 2 or reduced.shape[1] != 2:
        raise ValueError(f"Reduced combined embeddings must be 2D for UMAP visualization. Expected 2D embeddings, got shape {reduced.shape}")
    if len(modality_labels) != reduced.shape[0]:
        raise ValueError(f"Label length {len(modality_labels)} does not match the number of embeddings {combined_embeds.shape[0]}")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=modality_labels,cmap='coolwarm', alpha=0.7)

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.8)


    
    plt.title(title)
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticks([0, 1])  # Set colorbar ticks
    cbar.set_ticklabels(['Image Embeddings','Text Embeddings'])  # Set colorbar labels
    plt.grid(True)
    plt.tight_layout()
    plt.show()   
# Plot the embeddings
plot_embeddings(
    image_embeds=image_embeds.cpu().numpy(),
    text_embeds=text_embeds.cpu().numpy(),
    labels= image_labels + text_labels,  # Combine text labels and captions for unique labels
    modality_labels= [0] * len(image_embeds) + [1] * len(text_embeds),  # 0 for image embeddings, 1 for text embeddings
    title="COCO Image and Text Embeddings in 2D UMAP Space"
) 

fig = go.Figure(data=[
    go.Bar(name='Image Embeddings', x=image_labels, y=image_embeds.cpu().numpy().mean(axis=1)),
    go.Bar(name='Text Embeddings', x=first_captions, y=text_embeds.cpu().numpy().mean(axis=1))

])
fig.update_layout(
    barmode='group',
    title='Text and Image Embeddings',
    xaxis_title='Labels',
    yaxis_title='Embeddings'
)
fig.show()


# %%
# Cosine similarities for embeddings between each corresponding image and text embeddings (inter-modality)
cosine_similarities = t.cosine_similarity(image_embeds, text_embeds, dim = 1)
print(f"Cosine similarities in the embedding space: {cosine_similarities}")
print(f"Min cosine similarity: {cosine_similarities.min()}")
print(f"Max cosine similarity: {cosine_similarities.max()}")

# %%
# Convert the tensor to a numpy array for plotting

import plotly.express as px
cosine_similarities_np = cosine_similarities.detach().cpu().numpy()

fig = px.bar(
    x = text_labels,
    y = cosine_similarities_np,
    title = "Cosine Similarities between Image and Text Embeddings",
    labels = {'x': 'Text Labels', 'y': 'Cosine Similarities'},
)
fig.show()
# %%
# Similarity matrix for all image-text pairs (inter-modality)
similarity_matrix = t.nn.functional.cosine_similarity(
    image_embeds.unsqueeze(1),  # [batch, 1, d_model]
    text_embeds.unsqueeze(0),   # [1, batch, d_model]
    dim=-1                      # Result: [batch, batch]
)
similarity_matrix_np = similarity_matrix.detach().cpu().numpy()


fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix_np,  # Reshape to 2D for heatmap
                x=text_labels,
                y= image_labels, 
                colorscale='Viridis',
                colorbar=dict(title='Cosine Similarity'),
                zmin=0,zmax=1,  # Set the color scale range
                hoverongaps=False,
                hovertemplate='%{x}: %{y}<br>Cosine Similarity: %{z:.2f}',
                ))
fig.update_layout(
    title="All Pairwise Cosine Similarities Heatmap",
    xaxis_title='Captions',
    yaxis_title='Images',
    autosize=False,
    width=800,
    height=600,
    margin=dict(l=100, r=100, t=100, b=100)  # Adjust margins
 )
fig.show()
# similarity_range = cosine_similarities.max() - cosine_similarities.min()
similarity_range = cosine_similarities.max() - cosine_similarities.min()
average_similarity = cosine_similarities.mean()
print(f"Similarity range: {similarity_range}")
print(f"Average similarity: {average_similarity}")
if similarity_range < 0.1 and average_similarity < 0.8:
  print("Poor alignment")

# %%
# Cross-modal cosine similarity matrix (after projection)
# Cosine similarity between projected text and image features
with t.no_grad():
    text_outputs = model.text_model(**text_inputs)
    text_pooled_output = text_outputs.pooler_output
    image_outputs = model.vision_model(**image_inputs)
    image_pool_outputs = image_outputs.pooler_output
text_projected = model.text_projection(text_pooled_output) # [batch, d_model]
image_projected = model.visual_projection(image_pool_outputs) # [batch, d_model]
projected_cosine_similarities = f.cosine_similarity( 
    image_projected.unsqueeze(1) ,  # [batch, 1, d_model]
    text_projected.unsqueeze(0),   # [1, batch, d_model]
    dim=-1                      # Result: [batch, batch]
)
print(f"Projected cosine similarities in the embedding space: {projected_cosine_similarities}")
print (f"Min projected cosine similarity: {projected_cosine_similarities.min()}")
print(f"Max projected cosine similarity: {projected_cosine_similarities.max()}")
print(f" Average projected cosine similarity: {projected_cosine_similarities.mean()}")
go.Figure(data=go.Heatmap(
    z=projected_cosine_similarities.detach().cpu().numpy(),  # Reshape to 2D for heatmap
    x=text_labels,
    y=image_labels, 
    colorscale='Viridis',
    colorbar=dict(title='Projected Cosine Similarity'),
    zmin=0,zmax=1,  # Set the color scale range
    hoverongaps=False,
    hovertemplate='%{x}: %{y}<br>Cosine Similarity: %{z:.2f}',
)).update_layout(
    title="Projected Pairwise Cosine Similarities Heatmap",
    xaxis_title='Captions',
    yaxis_title='Images',
    autosize=False,
    width=800,
    height=600,
    margin=dict(l=100, r=100, t=100, b=100)  # Adjust margins
 ).show()
# %%
# Class names for the images
# Assuming the first caption corresponds to the first image in the batch
for i,label_indx in enumerate(first_captions):
   class_name = first_captions[i]
   print(f" Image {i}: class name = {class_name}")


# %%
# Finding the residual directions
# For Images (ViT/CLIP): the residual directions for a class in the column of the classification head weight matrix for that class. 
# For texts (CLIP): the residual direction for a label is the embedding for that label 
normalised_image_residual_directions = f.normalize(model.visual_projection.weight,dim=-1)  # [num_classes, d_model]
print(f"Visual residual directions: {normalised_image_residual_directions.shape}")

normalised_text_residual_directions = f.normalize(model.text_projection.weight,dim=-1)  # [num_classes, d_model]
print(f"Text residual directions: {normalised_text_residual_directions.shape}") 

# %%
# Access the hidden states

vision_hidden_states = outputs.vision_model_output.hidden_states
text_hidden_states = outputs.text_model_output.hidden_states

print("\n--- Vision Model Hidden States ---")
print(f"Type: {type(vision_hidden_states)}")  # This will be a tuple
print(f"Number of layers (including embeddings): {len(vision_hidden_states)}")

# Get the hidden states from the last layer of the vision model
last_layer_vision_hidden_states = vision_hidden_states[-1]
print(f"Shape of the last layer hidden states: {last_layer_vision_hidden_states.shape}")

print("\n--- Text Model Hidden States ---")
print(f"Type: {type(text_hidden_states)}")  # This will be a tuple
print(f"Number of layers (including embeddings): {len(text_hidden_states)}")

# Get the hidden state from the last layer of the text model
last_layer_text_hidden_states = text_hidden_states[-1]
print(f"Shape of the last layer hidden states: {last_layer_text_hidden_states.shape}")

# Get the classification token at the beginning
cls_vision_residual_stream = last_layer_vision_hidden_states[:, 0, :]
cls_text_residual_stream = last_layer_text_hidden_states[:, 0, :]
print(f"Shape of the CLS token in vision model: {cls_vision_residual_stream.shape}")
print(f"Shape of the CLS token in text model: {cls_text_residual_stream.shape}")

# Get CLIP's LayerNorm
vision_ln = model.vision_model.post_layernorm
text_ln = model.text_model.final_layer_norm

# Apply the LayerNorm to the residual streams
scaled_cls_vision = f.normalize(vision_ln(cls_vision_residual_stream),dim=-1)  # Normalize the CLS token for vision
scaled_cls_text = f.normalize(text_ln(cls_text_residual_stream),dim=-1)  # Normalize the CLS token for text

# Print the shapes of the scaled classification tokens
print(f"Scaled CLS Vision shape: {scaled_cls_vision.shape}")
print(f"Scaled CLS Text shape: {scaled_cls_text.shape}")

# Compute logits for each image-text pair in the batch
# image_residual_directions: [batch, d_model] or [d_model]
# text_residual_directions: [batch, d_model] or [d_model]
# If you want to compute the logit for each image-text pair, use einsum over the batch
def compute_dla_logits(scaled_cls_vision, scaled_cls_text, normalised_image_residual_directions, normalised_text_residual_directions):
    """
    Compute the logits for each image-text pair in the batch.
    Args:
        scaled_cls_vision: Scaled CLS token for vision.
        scaled_cls_text: Scaled CLS token for text.
        normalised_image_residual_directions: Normalized residual directions for images.
        normalised_text_residual_directions: Normalized residual directions for text.
    Returns:
        calculated_vision_logit: Logits for vision.
        calculated_text_logit: Logits for text.
    """
    calculated_vision_logit = einsum('b d,c d-> b c', scaled_cls_vision, normalised_image_residual_directions)  # Each image projected onto each of the 512 class directions
    calculated_text_logit = einsum('b d, c d -> b c', scaled_cls_text, normalised_text_residual_directions)  # Each image projected onto each of the 512 class directions
    print(f"Vision logit shape: {calculated_vision_logit.shape}")
    print(f"Text logit shape: {calculated_text_logit.shape}")
    return calculated_vision_logit, calculated_text_logit
vision_logit, text_logit = compute_dla_logits(
    scaled_cls_vision,
    scaled_cls_text,
    normalised_image_residual_directions,
    normalised_text_residual_directions
)

# Add the bias for each class to the vision and text logits if present
if getattr(model.visual_projection, 'bias', None) is not None:
    assert model.visual_projection.bias.shape == vision_logit.shape
    vision_logit += model.visual_projection.bias
else:
    print("No bias in the visual projection layer.")

if getattr(model.text_projection, 'bias', None) is not None:
    assert model.text_projection.bias.shape == text_logit.shape
    text_logit += model.text_projection.bias
else:
    print("No bias in the text projection layer.")

# Print the logits for the batch
# Batched retrieval with 8 images and 8 prompts
for i in range(len(vision_logit)):
    for j in range(len(text_logit)):
    
        print(f"Image {i}, Class {j}: Vision Logit: {round(vision_logit[i][j].item(),3)}, Text Logit: {round(text_logit[j][i].item(),3)}", # image i projected to prompt j direction
              f" Vision logit from model: {round(image_logits[i][j].item(),3)}, Text logit from model: {round(text_logits[j][i].item(),3)}") # prompt j projected to image i direction

pio.renderers.default = "browser"
import plotly.graph_objs as go

sim_matrix = t.mm(f.normalize(scaled_cls_text, dim=-1), f.normalize(scaled_cls_text, dim=-1).T)  # Cosine similarity matrix
print("Text CLS cosine similarity matrix :\n", sim_matrix)
# Prepare unique labels for each image

# Convert logits to numpy for plotting
vision_logit_np = vision_logit.detach().cpu().numpy()
text_logit_np = text_logit.detach().cpu().numpy()
vision_logit_diag = [vision_logit_np[i,i] for i in range(len(vision_logit_np))]  # Image i logit for prompt i
text_logit_diag = [text_logit_np[i,i] for i in range(len(text_logit_np))]  # prompt i logit for image i
# Create a grouped bar chart for direct logit attributions
fig = go.Figure(data=[
    go.Bar(name='Vision Logit', x=image_labels, y=vision_logit_diag),
    go.Bar(name='Text Logit', x=text_labels, y=text_logit_diag)
])

fig.update_layout(
    barmode='group',
    title='Direct Logit Attributions per Image',
    xaxis_title='Text Labels (Image Index)',
    yaxis_title='Logit Value'
)

fig.show()


# %%
# Investigate preprocessed tokens 
# Freeze the tokenizer pipeline
logit_scale = model.logit_scale

logits = logit_scale * image_projected @ text_projected.T
print(logits)

# %%
# Logit-level diagnostic matrix
sns.heatmap(logits.detach().cpu().numpy(), cmap='viridis')
plt.title("Text-Image Inputs")
plt.xlabel("Text Inputs")
plt.ylabel("Image Inputs")
plt.show()


# %%
# Average projected text and image features
print("Average projected text features:", text_projected.mean(dim=0))
print("Average projected image features:", image_projected.mean(dim=0))

# Cosine similarity between average projected text and image features
avg_projected_cosine_similarity = f.cosine_similarity(
    image_projected.mean(dim=0, keepdim=True),  # [1, d_model]
    text_projected.mean(dim=0, keepdim=True),   # [1, d_model]
    dim=-1                      # Result: [1]
)
print(f"Average projected cosine similarity: {round(avg_projected_cosine_similarity.item(),3)}")

# L2 distance between average projected text and image features
avg_projected_l2_distance = t.norm(
    image_projected.mean(dim=0)- text_projected.mean(dim=0)
)
print(f"Average projected L2 distance: {round(avg_projected_l2_distance.item(),3)}")

# %%
# Check for LayerNorm
vis_projection = model.visual_projection
text_projection = model.text_projection
for name,module in vis_projection.named_modules():
    if isinstance(module, t.nn.LayerNorm):
        print(f"LayerNorm found in visual projection: {name} with normalised shape {module.normalized_shape}")
    else:
        print(f"LayerNorm not found in visual projection")
for name,module in text_projection.named_modules():
    if isinstance(module, t.nn.LayerNorm):
        print(f"LayerNorm found in text projection: {name} with normalised shape {module.normalized_shape}")
    else:
        print(f"LayerNorm not found in text projection")

with t.no_grad():
    text_outputs = model.text_model(**text_inputs)
    text_pooled_output = text_outputs.pooler_output
    image_outputs = model.vision_model(**image_inputs)
    image_pool_outputs = image_outputs.pooler_output
    projected_vision_embeds = model.text_projection(text_pooled_output) # [batch, d_model]
    projected_text_embeds = model.visual_projection(image_pool_outputs) # [batch, d_model]
    text_sd = projected_text_embeds.std()
    vision_sd = projected_vision_embeds.std()
    text_mean = projected_text_embeds.mean()
    vision_mean = projected_vision_embeds.mean()
    print(f"Text projection standard deviation: {text_sd.item():.4f}")
    print(f"Vision projection standard deviation: {vision_sd.item():.4f}")
    print(f"Text projection mean: {text_mean.item():.4f}")
    print(f"Vision projection mean: {vision_mean.item():.4f}")

# Check norm inflation
# Check weight statistics
text_projected_weight = model.text_projection.weight
vision_projected_weight = model.visual_projection.weight
#print(f"Text projection weight shape: {text_projected_weight.shape}")
#print(f"Vision projection weight shape: {vision_projected_weight.shape}")
norm_text_projected_weight = t.norm(text_projected_weight, dim=1)  # L2 norm over the last dimension
norm_vision_projected_weight = t.norm(vision_projected_weight, dim=1)
print(f"Mean text projection weight norm: {norm_text_projected_weight.mean().item():.4f}")
print (f"Max text projection weight norm: {norm_text_projected_weight.max().item():.4f}")
print(f"Mean vision projection weight norm: {norm_vision_projected_weight.mean().item():.4f}")
print (f"Max vision projection weight norm: {norm_vision_projected_weight.max().item():.4f}")
# Check norm over training
text_projected_norm = t.norm(projected_text_embeds, dim=-1)  # L2 norm over the last dimension
vision_projected_norm = t.norm(projected_vision_embeds, dim=-1)
print(f"Mean text projected norm: {text_projected_norm.mean().item():.4f}")
print(f"Mean vision projected norm: {vision_projected_norm.mean().item():.4f}")
# Check feature dominance 
var_text_dimwise = t.var(projected_text_embeds, dim=0) # Variance along the text feature dimension
var_visual_dimwise = t.var(projected_vision_embeds, dim=0) # Variance along the visual feature dimension
text_top_k = t.topk(var_text_dimwise, k=10) # Get the top 10 features with highest variance
var_visual_dimwise = t.var(projected_vision_embeds, dim=0) # Variance along the feature dimension
vision_top_k = t.topk(var_visual_dimwise, k=10) # Get the
rounded_text_top_k = text_top_k.values.round(decimals=4).cpu().numpy()  # Round the values to 4 decimal places
rounded_vision_top_k = vision_top_k.values.round(decimals=4).cpu().numpy()  # Round the values to 4 decimal places
print(f"Top 10 text dominant dimensions: {text_top_k.indices.tolist()}")
print (f"Top 10 text values : {rounded_text_top_k}")
print(f"Top 10 vision dominant dimensions: {vision_top_k.indices.tolist()}")
print (f"Top 10 vision values : {rounded_vision_top_k}")

# %%
# Pre-projected text and image embeddings' L2 norms
pre_text_l2_norms = t.norm(text_embeds, dim=-1)
pre_vision_l2_norms = t.norm(image_embeds, dim=-1)

# Post-projected text and image embeddings' L2 norms
post_text_l2_norms = t.norm(projected_text_embeds, dim=-1)
post_vision_l2_norms = t.norm(projected_vision_embeds, dim=-1)

# Print the L2 norms
print(f"Pre-projected mean text L2 norms: {pre_text_l2_norms.mean().item():0.4f}\n Pre-projected  mean vision L2 norms: {pre_vision_l2_norms.mean().item():0.4f}")
print(f"Post-projected mean text L2 norms: {post_text_l2_norms.mean().item():0.4f}\n Post-projected mean vision L2 norms: {post_vision_l2_norms.mean().item():0.4f}")

# %%
def normalise_l2(name, text_projected, vision_projected):
    """
    Normalise the L2 norms of the projected text and vision embeddings.
    Args:
        name: Name of the model.
        text_projected: Projected text embeddings.
        vision_projected: Projected vision embeddings.
    Returns:
        None
    """
    print(f"Normalising L2 norms for {name}...")
    text_norm = t.norm(text_projected, dim=-1, keepdim=True)
    vision_norm = t.norm(vision_projected, dim=-1, keepdim=True)
    text_projected /= text_norm
    vision_projected /= vision_norm
    print(f"Normalised text projected shape: {text_projected.shape}")
    print(f"Normalised vision projected shape: {vision_projected.shape}")
    return text_projected, vision_projected
normalised_text_projected, normalised_vision_projected = normalise_l2('CLIP', projected_text_embeds, projected_vision_embeds) 
plot_embeddings(
    image_embeds=normalised_vision_projected.cpu().numpy(),
    text_embeds=normalised_vision_projected.cpu().numpy(),
    labels=image_labels + text_labels,  # Combine text labels and captions for unique labels
    modality_labels=[0] * len(normalised_vision_projected) + [1] * len(normalised_text_projected),  # 0 for image embeddings, 1 for text embeddings
    title="Normalised CLIP Image and Text Embeddings in 2D UMAP Space"
    
)

# %%
print(f"Hooked ConViT Congfig", HookedViTConfig())
print(f"CLIP model config from clip_model", clip_model.cfg)
print(f"CLIP model config from CLIPModel", CLIPConfig())
print(f"Hooked ConViT Config's patch size: {HookedViTConfig().patch_size}")
print(f"CLIP model config patch size: {clip_model.cfg.patch_size}")
print(f"Hooked ConViT Config's image size: {HookedViTConfig().image_size}")
print(f"CLIP model config image size: {clip_model.cfg.image_size}")



