import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.transforms import transforms

from vit_prisma.models.weight_conversion import hf_hub_download
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.utils.constants import DEVICE
from vit_prisma.utils.enums import ModelType
from vit_prisma.utils.constants import (
    DATA_DIR,
    MODEL_DIR,
)


def load_remote_sae_and_model(
    repo_id: str,
    checkpoint="n_images_2600058.pt",
    config_file: str = "config.json",
    current_cfg: dict = None,
    model_type: ModelType = ModelType.VISION,
) -> Tuple[SparseAutoencoder, HookedViT]:
    """Load and test SAE from HuggingFace."""

    print(f"Loading SAE from repo_id: {repo_id} with checkpoint: {checkpoint}")
    sae_path = hf_hub_download(repo_id, checkpoint)
    sae_config_path = hf_hub_download(repo_id, config_file)

    sae = SparseAutoencoder.load_from_pretrained(
        sae_path, config_path=sae_config_path, current_cfg=current_cfg
    )

    print(f"Loading model name: {sae.cfg.model_name}")
    print(f"The config device is: {sae.cfg.device}")

    if model_type == ModelType.VISION:
        model = HookedViT.from_pretrained(
            sae.cfg.model_name, is_timm=False, is_clip=True
        ).to(sae.cfg.device)
    else:
        model = HookedTextTransformer.from_pretrained(
            sae.cfg.model_name, is_timm=False, is_clip=True
        ).to(sae.cfg.device)

    sae = sae.to(DEVICE)
    print(f"Using device: {DEVICE}")

    return sae, model


def load_clip_models(layer: int):
    """Load a Prisma SAE and corresponding CLIP model for the given layer."""

    repo_id = f"Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-{layer}-hook_mlp_out-l1-1e-05"
    sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints") + f"/{repo_id}"
    os.makedirs(sae_path, exist_ok=True)

    overload_cfg = {
        # Data
        "dataset_path": str(DATA_DIR / "imagenet"),
        "dataset_train_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
        "dataset_val_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        # Logging
        "checkpoint_path": str(MODEL_DIR / "sae/imagenet/checkpoints"),
        "wandb_log_frequency": 100,
        # "wandb_entity": "<FILL IN>",
        "wandb_project": "imagenet",
        "log_to_wandb": False,
        "verbose": True,
        # SAE config
        "sae_path": sae_path,
    }

    return load_remote_sae_and_model(repo_id, current_cfg=overload_cfg)


def calculate_clean_accuracy(
    net: HookedViT,
    classifier: torch.Tensor,
    data_loader,
    device=DEVICE,
    top_k: int = 1,
    sae: SparseAutoencoder = None,
):
    """Calculate the top k clean accuracy of a CLIPmodel on a dataset."""

    net.eval()
    correct = 0
    total = 0

    for batch in tqdm(data_loader):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        with torch.no_grad():
            logits = sae.get_test_loss(images, net) if sae else net(images)
            logits = 100.0 * logits @ classifier
            preds = logits.topk(top_k)[1].t()[0]
            correct += preds.eq(labels).sum().item()
            total += len(labels)

        torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0
    return accuracy, total


def plot_image(image, unstandardise=True):
    plt.figure()
    plt.axis("off")

    if unstandardise:
        print("Unstandardising image")
        get_inverse = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = get_inverse(image)

    image = image.permute(1, 2, 0)
    plt.imshow(image, vmin=-2.5, vmax=2.0)


def get_feature_activations(model_input, model, sae):
    # Run the batch through the model to get activations
    _, cache = model.run_with_cache(model_input, names_filter=sae.cfg.hook_point)
    hook_point_activation = cache[sae.cfg.hook_point].to(DEVICE)

    # Calculate the SAE features and stats for the batch
    _, feature_acts, *_ = sae(hook_point_activation)

    return feature_acts


def plot_act_distribution(
    tensor, n_top=10
):
    """Plot feature distribution & return top indices/values."""
    if tensor.is_cuda or tensor.device.type == "mps":
        tensor = tensor.detach().cpu()
    data = tensor.detach().numpy()

    threshold = 0.01
    x_indices = [j for j, val in enumerate(data) if val > threshold]
    y_values = [val for val in data if val > threshold]

    df = pd.DataFrame({'index': x_indices, 'value': y_values})
    fig = px.bar(df, x='index', y='value', width=1200, height=400)

    top_indices = np.argsort(data)[-n_top:]
    top_values = data[top_indices]
    print(f"Top {n_top} feature indices: {[v.item() for v in top_indices]}")
    print(f"Top {n_top} feature values: {top_values}")

    fig.add_trace(
        go.Scatter(
            x=top_indices, y=top_values, mode='markers+text',
            marker=dict(color='red', size=8),
            text=[f"{idx}" for idx in top_indices],
            textposition="top center", textfont=dict(size=8),
            showlegend=False
        )
    )

    fig.update_layout(
        title=f"Feature Activations",
        xaxis_title="Feature Index", yaxis_title="Feature Value",
        plot_bgcolor='rgba(255,255,255,1)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.3)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.3)')
    )
    fig.show()
    return top_indices, top_values


def plot_imgs_for_one_feature(
    feature_idx, imagenet_indices, activation_values, viz_data, cfg
):
    """Plot top activating images for a single feature."""
    grid_size = int(np.ceil(np.sqrt(len(imagenet_indices))))
    fig, axs = plt.subplots(int(np.ceil(len(imagenet_indices) / grid_size)), grid_size, figsize=(15, 15))
    fig.suptitle(f"Layer: {cfg.hook_point}, Feature: {feature_idx}")

    axs = axs.flatten()
    for i, (image_idx, image_act_value) in enumerate(zip(imagenet_indices, activation_values)):
        image_tensor, _, _ = viz_data[image_idx]
        display = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        axs[i].imshow(display)
        axs[i].set_title(f"Img idx: {image_idx} Act: {image_act_value.item():.3f}")
        axs[i].axis("off")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_top_imgs_for_features(
    top_indices, ref_imgs_per_feat, viz_data, sae, top_k=10
):
    """Plot top activating images for a list of feature indices."""
    
    indices_to_plot = top_indices[-top_k:]
    for i, feature_idx in enumerate(indices_to_plot):
        v = ref_imgs_per_feat[feature_idx]
        vals, imgs = v["values"], v["indices"]
        print(f"Feature {feature_idx} (Rank {len(top_indices)-len(indices_to_plot)+i}): Top Images: {imgs}, Activations: {vals}")
        plot_imgs_for_one_feature(feature_idx, imgs, vals, viz_data, sae.cfg)
