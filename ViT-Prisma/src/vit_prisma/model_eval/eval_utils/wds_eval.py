"""Evaluate on standard classification webdatasets."""

import logging
import os

import open_clip
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from sklearn.metrics import balanced_accuracy_score
from vit_prisma.models.model_loader import _get_open_clip_config, load_hooked_model
from vit_prisma.models.weight_conversion import remove_open_clip_prefix
from vit_prisma.utils.enums import ModelType


# Suppress PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


# create the wrapper model using open_clip and sae
def create_model_for_prisma(model_arch, model_path, model_kwargs=None, sae_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)

    if model_kwargs is None:
        model_kwargs = {}

    # Check if model_path is a local checkpoint
    if os.path.exists(model_path):
        # Load from local checkpoint with local_path parameter
        model = load_hooked_model(
            model_name=model_arch,
            model_type=ModelType.VISION,
            device=device,
            pretrained=True,
            local_path=model_path,
            **model_kwargs,
        )
    else:
        # Load pretrained model
        model = load_hooked_model(
            model_name=model_arch,
            model_type=ModelType.VISION,
            device=device,
            pretrained=True,
            **model_kwargs,
        )

    # Ensure the ViT-Prisma model is properly moved to device
    model = model.to(device)
    model.eval()

    # Also load the original OpenCLIP model for text encoding
    text_encoder = None
    if os.path.exists(model_path):
        # Load from local checkpoint with local_path parameter
        text_encoder = load_hooked_model(
            model_name=model_arch,
            model_type=ModelType.TEXT,
            device=device,
            pretrained=True,
            local_path=model_path,
            **model_kwargs,
        )
    else:
        # Load pretrained model
        text_encoder = load_hooked_model(
            model_name=model_arch,
            model_type=ModelType.TEXT,
            device=device,
            pretrained=True,
            **model_kwargs,
        )

    text_encoder.eval()
    text_encoder = text_encoder.to(device)
    print("Loaded original OpenCLIP model for text encoding")

    # Load and integrate SAE if provided
    if sae_path and os.path.exists(sae_path):
        print(f"Loading SAE from: {sae_path}")
        from vit_prisma.sae.sae import SparseAutoencoder

        # Load SAE
        sae = SparseAutoencoder.load_from_pretrained(weights_path=sae_path)
        sae.to(device)
        sae.eval()

        print(f"SAE loaded successfully:")
        print(f"  Hook point: {sae.cfg.hook_point}")
        print(f"  Dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
        print(f"  Architecture: {sae.cfg.architecture}")

        # Create a wrapper model that integrates CLIP with SAE
        model = CLIPWithSAEWrapper(model, sae, sae.cfg.hook_point, text_encoder)
        print("CLIP model integrated with SAE")
    elif text_encoder is not None:
        # Even without SAE, wrap the model so it's still functional
        model = CLIPWithSAEWrapper(model, None, None, text_encoder)
        print("ViT-Prisma model wrapped with OpenCLIP for text encoding")

    # Ensure the wrapper is properly moved to device
    model = model.to(device)

    # Get transform - try to use model's transform if available
    if hasattr(model, "transform"):
        transform = model.transform
    else:
        # Fallback to standard transform
        from torchvision import transforms

        class GrayscaleToRGB:
            """Convert grayscale images to RGB by repeating the channel 3 times."""

            def __call__(self, tensor):
                if tensor.shape[0] == 1:  # Grayscale image
                    return tensor.repeat(3, 1, 1)
                return tensor

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                GrayscaleToRGB(),  # Convert grayscale to RGB
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    return model, transform, device


class CLIPWithSAEWrapper(torch.nn.Module):
    """Wrapper class that integrates CLIP model with SAE."""

    def __init__(self, vision_encoder, sae, hook_point, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.sae = sae
        self.hook_point = hook_point
        self.use_sae = True
        self._hook_handle = None
        self.text_encoder = text_encoder  # For text encoding

        # Register the hook once during initialization for evaluation
        self._register_hook()

    def _find_module_by_name(self, model, target_name):
        """Find a module by its name in the model hierarchy."""
        for name, module in model.named_modules():
            if name == target_name:
                return module
        return None

    def _sae_hook_fn(self, module, input, output):
        """Hook function that applies SAE to the activations."""
        if self.use_sae and self.sae is not None:
            # Apply SAE to the output activations
            sae_out, feature_acts = self.sae.encode(output)
            return self.sae.decode(feature_acts)
        return output

    def _register_hook(self):
        """Register the SAE hook on the target module."""
        if self._hook_handle is not None:
            return  # Hook already registered

        # Skip hook registration if no SAE or hook point
        if self.sae is None or self.hook_point is None:
            print("No SAE or hook point specified, skipping hook registration")
            return

        target_module = self._find_module_by_name(self.vision_encoder, self.hook_point)
        if target_module is None:
            print(f"Warning: Could not find module '{self.hook_point}' in the model")
            print("Available modules:")
            for name, _ in self.vision_encoder.named_modules():
                if "block" in name.lower() or "layer" in name.lower():
                    print(f"  {name}")
            return

        self._hook_handle = target_module.register_forward_hook(self._sae_hook_fn)
        print(f"SAE hook registered at: {self.hook_point}")

    def _remove_hook(self):
        """Remove the registered hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, x):
        # Hook is already registered, just do forward pass
        return self.vision_encoder(x)

    def encode_image(self, images, normalize=True):
        # Use forward pass with SAE (hook is already active)
        features = self.forward(images)

        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def encode_text(self, text, normalize=True):
        features = self.text_encoder(text)

        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def to(self, device):
        self.vision_encoder = self.vision_encoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        if self.sae is not None:
            self.sae = self.sae.to(device)
        return self

    def eval(self):
        self.vision_encoder.eval()
        self.text_encoder.eval()
        if self.sae is not None:
            self.sae.eval()
        return self

    def train(self, mode=True):
        raise NotImplementedError("CLIPWithSAEWrapper is for evaluation only.")

    def __del__(self):
        """Cleanup hook when object is destroyed."""
        self._remove_hook()


def get_text_tokenizer(
    model_name: str,
    local_path: str = None,
    **kwargs,
):

    if local_path:
        # Load the raw config dictionary instead of the processed HookedViTConfig
        old_config = _get_open_clip_config(
            model_name, ModelType.VISION, local_path=local_path
        )

        text_config = old_config.get("text_cfg", {})
        if "tokenizer_kwargs" in text_config:
            tokenizer_kwargs = dict(text_config["tokenizer_kwargs"], **kwargs)
        else:
            tokenizer_kwargs = kwargs

        DEFAULT_CONTEXT_LENGTH = 77
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

        model_name = model_name.lower()
        if text_config.get("hf_tokenizer_name", ""):
            raise NotImplementedError
        elif "siglip" in model_name:
            raise NotImplementedError
        else:
            from open_clip.tokenizer import SimpleTokenizer

            # TODO: currently all Prisma pretrained openCLIP SAEs use the same text tokenizer, so this is fine
            # need to be fixed if eval on new models with different tokenizers
            tokenizer = SimpleTokenizer(
                context_length=context_length,
                **tokenizer_kwargs,
            )
    else:
        import open_clip

        # TODO: currently all Prisma pretrained openCLIP SAEs use the same text tokenizer, so this is fine
        # need to be fixed if eval on new models with different tokenizers
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return tokenizer


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=8,
    return_preds=False,
    return_topk=False,
    model_kwargs=None,
    use_vit_prisma=False,
    sae_path=None,
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = create_model_for_prisma(
        model_arch,
        model_path,
        model_kwargs,
        sae_path=sae_path,
    )

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    # Get appropriate tokenizer

    local_path = model_path if os.path.exists(model_path) else None
    tokenizer = get_text_tokenizer(model_arch, local_path=local_path)

    classifier = zsc.zero_shot_classifier(
        model,
        tokenizer,
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target, l0, l1, l2 = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )
    with torch.no_grad():
        pred = logits.argmax(axis=1).cpu()
        target = target.cpu()

    # Compute metrics
    if len(dataset.classes) >= 5:
        acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    else:
        (acc1,) = zsc.accuracy(logits, target, topk=(1,))
        acc5 = None
    mean_per_class_recall = balanced_accuracy_score(target, pred)

    metrics = {
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
        "l0": l0.mean().numpy().item(),
        "l1": l1.mean().numpy().item(),
        "l2": l2.mean().numpy().item(),
    }

    if return_preds:
        if return_topk:
            with torch.no_grad():
                _, topk_pred = torch.topk(logits, int(return_topk), dim=1)
                topk_pred = topk_pred.cpu()
            return metrics, topk_pred, target
        return metrics, pred, target
    return metrics
