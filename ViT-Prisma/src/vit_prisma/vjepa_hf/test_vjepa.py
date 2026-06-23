from PIL import Image
import requests
from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAModel, VJEPAImageProcessor
from vit_prisma.vjepa_hf.configs.configs import CONFIGS
import yaml

import torch

from vit_prisma.models.base_vit import HookedViT
import pytest


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def check_matching(outputs, outputs_hooked, tolerance=1e-3):
    # Compare the outputs
    diff = torch.abs(outputs_hooked - outputs)
    max_diff = torch.max(diff)
    if max_diff < tolerance:
        print("Outputs match within tolerance")
    else:
        print(f"Outputs do not match within tolerance. Max difference: {max_diff.item()}")
    # Alternatively, you can use torch.allclose() function
    if torch.allclose(outputs_hooked, outputs, atol=tolerance):
        print("Outputs match within tolerance")
    else:
        print("Outputs do not match within tolerance")
    # You can also calculate the mean squared error (MSE) or mean absolute error (MAE)
    mse = torch.mean((outputs_hooked - outputs) ** 2)
    mae = torch.mean(torch.abs(outputs_hooked - outputs))
    print(f"MSE: {mse.item():.6f}")
    print(f"MAE: {mae.item():.6f}")

    diff = torch.abs(outputs_hooked - outputs)
    max_diff = torch.max(diff)
    # Get the indices of the elements that don't match
    print("number of total indices", len(diff.flatten()))
    non_match_indices = torch.nonzero(diff > tolerance)
    if non_match_indices.numel() > 0:
        print(f"Number of non-matching indices: {non_match_indices.shape[0]}")
        print("Non-matching indices:")
        # for i in range(non_match_indices.shape[0]):
        #     batch_idx, seq_idx, embed_idx = non_match_indices[i]
        #     print(f"Batch index: {batch_idx.item()}, Sequence index: {seq_idx.item()}, Embedding index: {embed_idx.item()}")
    else:
        print("All elements match within tolerance")

def run_imagenet():
    # run for original model


    return 

        
# 1. LOAD HOOKED MODEL
def test_hooked_model_vjepa_loading():
    og_model_name = "vjepa2-vitl-fpc64-256"
    print("Loading hooked model")
    hooked_model = HookedViT.from_pretrained(og_model_name, is_timm=False, is_clip=False, fold_ln=False) # in future, do all models
    assert hooked_model is not None, "Hooked model loading failed"
    assert hooked_model.config is not None, "Hooked model config is None"

# 2. LOAD ORIGINAL MODEL
def test_original_model_vjepa_loading():
    model_name = "vjepa_v1_vit_huge"
    config = CONFIGS["v1"]["vit_h"]
    model_paths = yaml.safe_load(open('paths_cw.yaml'))
    model_path = model_paths[model_name]["loc"]
    model = VJEPAModel.from_pretrained(model_path)
    assert model is not None, "Original model loading failed"
    assert model.config is not None, "Original model config is None"


# 3. LOAD INPUT FOR BOTH MODELS
def test_vjepa_input_preparation():
    config = CONFIGS["v1"]["vit_h"]
    image = prepare_img()
    processor = VJEPAImageProcessor(crop_size=config.crop_size)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times for now
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
    assert pixel_values is not None, "Pixel values preparation failed"
    print("Pixel values shape:", pixel_values.shape)

    # 4. FEED INTO BOTH MODELS
    # TODO: Write a test that feeds the pixel_values into both the hooked model and the original model, and checks if the outputs match within a certain tolerance.
#def test_vjepa_model_outputs():
#    hooked_model = HookedViT.from_pretrained("vjepa2-vitl-fpc64-256", is_timm=False, is_clip=False, fold_ln=False)
#    model_paths = yaml.safe_load(open('paths_cw.yaml'))
#    model_path = model_paths["vjepa_v1_vit_huge"]["loc"]
#    model = VJEPAModel.from_pretrained(model_path)
    
#    outputs_hooked = hooked_model(pixel_values)
#    print(outputs_hooked)
#    print("Hooked outputs shape", outputs_hooked.shape)

 #   outputs = model(pixel_values)
#    print(outputs)
#    assert outputs[0] is not None, "Outputs preparation failed"
#    print("Outputs shape:", outputs[0].shape)

# 5. CHECK IF OUTPUTS MATCH
#check_matching(outputs[0], outputs_hooked)

# 6. RUN IMAGENET
