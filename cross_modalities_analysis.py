# %%
import vit_prisma
from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils import prisma_utils

#from transformers import CLIPProcessor, CLIPModel

from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word
import numpy as np
import torch as t
from fancy_einsum import einsum
from collections import defaultdict

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.colors as mcolors

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from IPython.display import display, HTML


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
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# Obtain the image
image = Image.open('cat_dog.jpeg')
image = transform(image)
plot_image(image)

# %%
# Loading the image
device = "cuda" if t.cuda.is_available() else "cpu"
model = model.to(device)
text_labels = ["a photo of a cat", "a photo of a dog"]
text_inputs = processor(text=text_labels, return_tensors="pt").to(device)
image_inputs = processor(images=image, return_tensors="pt")
image_inputs["pixel_values"] = image_inputs["pixel_values"].to(device)

with t.no_grad():
    outputs = model(**image_inputs, **text_inputs,output_hidden_states=True)
    logits = outputs.logits_per_image
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

print("Logits:",logits,"Label probs:", probs)

# %%
from vit_prisma.visualization.visualize_image import display_grid_on_image

display_grid_on_image(image, patch_size=32)

# %%
#from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word
#from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
#from vit_prisma.utils.prisma_utils import test_prompt




# %%
# Find cosine similarities
cosine_similarities = outputs.logits_per_image
print(f"cosine similarities: {cosine_similarities}")

# %%
# Access the hidden states

vision_hidden_states = outputs.vision_model_output.hidden_states
text_hidden_states = outputs.text_model_output.hidden_states

print("\n--- Vision Model Hidden States ---")
print(f"Type: {type(vision_hidden_states)}") # This will be a tuple
print(f"Number of layers (including embeddings): {len(vision_hidden_states)}")

# Get the hidden states from the last layer of the vision model
last_layer_vision_hidden_states = vision_hidden_states[-1]
print(f"Shape of the last layer hidden states: {last_layer_vision_hidden_states.shape}")

print("\n--- Text Model Hidden States ---")
print(f"Type: {type(text_hidden_states)}") # This will be a tuple
print(f"Number of layers (including embeddings): {len(text_hidden_states)}")

# Get the hidden state from the last layer of the text model

last_layer_text_hidden_states = text_hidden_states[-1]
print(f"Shape of the last layer hidden states: {last_layer_text_hidden_states.shape}")

# %%
# Get the embeddings
image_feats = outputs.image_embeds
text_feats = outputs.text_embeds
print(f"image_feats: {image_feats.shape}")
print(f"text_feats: {text_feats.shape}")

# %%
# Cosine similarities for embeddings
cosine_similarities = t.cosine_similarity(image_feats, text_feats, dim = 1)
print(f"Cosine similarities: {cosine_similarities}")

# %%
# Convert the tensor to a numpy array for plotting

import plotly.express as px
cosine_similarities_np = cosine_similarities.detach().cpu().numpy()

fig = px.bar(
    x = text_labels,
    y = cosine_similarities_np,
    title = "Cosine Similarities between Image and Text Features",
    labels = {'x': 'Text Labels', 'y': 'Cosine Similarities'},

)
fig.show()


# similarity_range = cosine_similarities.max() - cosine_similarities.min()
similarity_range = cosine_similarities.max() - cosine_similarities.min()
if similarity_range < 0.1:
  print("Poor alignment")