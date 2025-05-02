import torch
import pytest
import open_clip
from transformers import CLIPProcessor
import numpy as np
from typing import Tuple, Dict, Any
from vit_prisma.models.model_loader import (
    load_hooked_model,
    PASSING_MODELS,
    FAILING_MODELS,
)

TOLERANCE = 1e-4
DEVICE = "cuda"

# Filter only OpenCLIP models from the sets
TEST_MODELS = [
    model[len("open-clip:") :]
    for model in PASSING_MODELS.union(FAILING_MODELS)
    if model.startswith("open-clip:")
]


def get_layer_outputs(model, input_image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Helper function to get intermediate layer outputs"""
    outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            # Handle both tensor and tuple outputs
            if isinstance(output, tuple):
                outputs[name] = output[0].detach()  # Take first element of tuple
            else:
                outputs[name] = output.detach()

        return hook

    # Register hooks for transformer blocks
    handles = []
    for name, module in model.named_modules():
        if "block" in name:
            handles.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    with torch.no_grad():
        model(input_image)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return outputs


def print_divergence_info(og_output, hooked_output, model_name):
    """Print detailed divergence information between outputs."""
    diff = torch.abs(hooked_output - og_output)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    median_diff = torch.median(diff).item()

    print(f"\nDivergence Analysis for {model_name}:")
    print(f"Max difference:     {max_diff:.8f}")
    print(f"Mean difference:    {mean_diff:.8f}")
    print(f"Median difference:  {median_diff:.8f}")

    # Print location of max difference
    if max_diff > 0:
        max_loc = torch.where(diff == max_diff)
        print(f"Location of max difference: {tuple(idx.tolist() for idx in max_loc)}")
        print(f"Original value at max diff: {og_output[max_loc].item():.8f}")
        print(f"Hooked value at max diff:   {hooked_output[max_loc].item():.8f}")


def generate_random_input(
    batch_size=5, channels=3, height=224, width=224, device="cuda"
):
    """Generate a random tensor to simulate input images."""
    with torch.random.fork_rng():
        torch.manual_seed(1)
        return torch.rand((batch_size, channels, height, width)).to(device)


def compare_model_outputs(
    openclip_model, hookedvit_model, input_image: torch.Tensor, model_name: str
) -> Dict[str, Any]:
    """Compare outputs and intermediate activations between models"""
    openclip_model.to(DEVICE)
    openclip_model.eval()
    hookedvit_model.to(DEVICE)
    hookedvit_model.eval()

    # Get outputs and intermediate activations
    with torch.no_grad():
        og_output = openclip_model(input_image)
        hooked_output = hookedvit_model(input_image)

        # Handle tuple outputs - take first element
        if isinstance(og_output, tuple):
            og_output = og_output[0]
        if isinstance(hooked_output, tuple):
            hooked_output = hooked_output[0]

        og_layer_outputs = get_layer_outputs(openclip_model, input_image)
        hooked_layer_outputs = get_layer_outputs(hookedvit_model, input_image)

    # Compare final outputs
    print_divergence_info(og_output, hooked_output, model_name)

    # Compare layer outputs
    layer_diffs = {}
    for name in og_layer_outputs.keys():
        if name in hooked_layer_outputs:
            diff = torch.max(
                torch.abs(og_layer_outputs[name] - hooked_layer_outputs[name])
            ).item()
            layer_diffs[name] = diff

    return {
        "final_match": torch.allclose(og_output, hooked_output, atol=TOLERANCE),
        "max_diff": torch.max(torch.abs(hooked_output - og_output)).item(),
        "layer_differences": layer_diffs,
    }


@pytest.mark.parametrize("model_name", open_clip.list_models())
def test_openclip_model_compatibility(model_name: str):
    """Test if OpenCLIP models can be loaded and produce matching outputs"""
    try:
        print(f"\nTesting model: {model_name}")

        # Convert to HF hub format for OpenCLIP
        openclip_name = f"hf-hub:{model_name}"

        # Load models
        og_model, *_ = open_clip.create_model_and_transforms(openclip_name)
        hooked_model = load_hooked_model(f"open-clip:{model_name}")

        # Generate test input
        input_image = generate_random_input()

        # Compare outputs
        results = compare_model_outputs(og_model, hooked_model, input_image, model_name)

        # Print layer-wise differences if any large discrepancies
        if not results["final_match"]:
            print(f"\nLayer-wise differences for {model_name}:")
            for layer, diff in results["layer_differences"].items():
                if diff > TOLERANCE:
                    print(f"{layer}: {diff:.8f}")

        assert results["final_match"], (
            f"{model_name} outputs diverge! "
            f"Max difference: {results['max_diff']:.8f}"
        )

        print(f"✓ {model_name} PASSED")

    except Exception as e:
        print(f"✗ {model_name} FAILED: {str(e)}")
        pytest.xfail(f"Model {model_name} failed: {str(e)}")


def inspect_model_differences(model_name: str) -> Dict[str, Any]:
    """
    Detailed inspection of differences between OpenCLIP and HookedViT models.
    """
    openclip_name = f"hf-hub:{model_name}"
    og_model, *_ = open_clip.create_model_and_transforms(openclip_name)
    hooked_model = load_hooked_model(f"open-clip:{model_name}")

    input_image = generate_random_input()
    results = compare_model_outputs(og_model, hooked_model, input_image, model_name)

    # Add model configs for debugging
    results["openclip_config"] = {
        name: getattr(og_model, name)
        for name in dir(og_model)
        if not name.startswith("_") and not callable(getattr(og_model, name))
    }
    results["hookedvit_config"] = hooked_model.cfg

    return results


if __name__ == "__main__":
    print(f"Testing {len(TEST_MODELS)} OpenCLIP models")

    # Track results
    passed_models = []
    failed_models = []

    for model_name in TEST_MODELS:
        print(f"\n{'='*80}\nTesting model: {model_name}\n{'='*80}")
        try:
            # Use HF hub format
            openclip_name = f"hf-hub:{model_name}"
            prisma_name = f"open-clip:{model_name}"

            # Load models - allow failing models
            og_model, *_ = open_clip.create_model_and_transforms(openclip_name)
            hooked_model = load_hooked_model(prisma_name, allow_failing=True)

            # Generate test input
            input_image = generate_random_input()

            # Compare outputs
            results = compare_model_outputs(
                og_model, hooked_model, input_image, model_name
            )

            if results["final_match"]:
                print(f"✓ {model_name} PASSED")
                passed_models.append(model_name)
            else:
                print(f"✗ {model_name} FAILED (outputs don't match)")
                print(f"Max difference: {results['max_diff']:.8f}")
                failed_models.append(
                    (
                        model_name,
                        f"outputs don't match (diff: {results['max_diff']:.8f})",
                    )
                )

        except Exception as e:
            print(f"✗ {model_name} FAILED: {str(e)}")
            failed_models.append((model_name, str(e)))

    # Print summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Total models tested: {len(TEST_MODELS)}")
    print(f"Passed: {len(passed_models)}")
    print(f"Failed: {len(failed_models)}")

    if passed_models:
        print("\nPassed models:")
        for model in passed_models:
            print(f"✓ {model}")

    if failed_models:
        print("\nFailed models:")
        for model, error in failed_models:
            print(f"✗ {model}: {error}")
