import logging

import open_clip
import torch

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_prisma.utils.constants import DEVICE

# from vit_prisma.models.build_zero_shot_classifier import build_zero_shot_classifier


def get_autocast():
    """
    Returns appropriate autocast context manager based on CUDA availability.

    Returns:
        callable: Either torch.cuda.amp.autocast if CUDA is available or a dummy context manager.
    """
    if torch.cuda.is_available():
        return autocast
    else:
        return DummyContextManager


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): The model output logits
        target (torch.Tensor): The ground truth labels
        topk (tuple): The values of k for which to compute the accuracy

    Returns:
        list: List of top-k accuracies for each k in topk
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, classifier, dataloader, device="cuda", fwd_hooks=None):
    """
    Run evaluation of the model on the provided dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate
        classifier (torch.Tensor): Zero-shot classifier weights
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        device (str): Device to run evaluation on (default: "cuda")
        fwd_hooks (list, optional): List of forward hooks to apply during model inference

    Returns:
        tuple: Top-1 and top-5 accuracy as floats
    """
    # autocast = get_autocast(args.precision)

    # input_dtype = get_input_dtype(args.precision)

    print(f"Running on {device}")
    model.to(device)

    with torch.inference_mode():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                if fwd_hooks is not None and hasattr(model, "run_with_hooks"):
                    # Run with hooks if provided and supported
                    output = model.run_with_hooks(images, fwd_hooks=fwd_hooks)
                else:
                    output = model(images)

                image_features = (
                    output["image_features"] if isinstance(output, dict) else output
                )
                if isinstance(image_features, tuple):
                    image_features = image_features[0]
                image_features = image_features.cpu()
                logits = 100.0 * image_features @ classifier
                logits = logits.to(device)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def zero_shot_eval(
    model,
    data,
    model_name,
    pretrained_classifier,
    tokenizer=None,
    fwd_hooks=None,
):
    """
    Performs zero-shot evaluation on ImageNet validation sets.

    Args:
        model (torch.nn.Module): Model to evaluate
        data (dict): Dictionary containing dataset objects
        model_name (str): Name of the model architecture
        pretrained_classifier (torch.Tensor, optional): Pre-computed zero-shot classifier
        tokenizer (callable, optional): Tokenizer for creating text embeddings
        fwd_hooks (list, optional): List of forward hooks to apply during model inference

    Returns:
        dict: Dictionary containing evaluation results (top-1 and top-5 accuracies)
    """
    if "imagenet-val" not in data and "imagenet-v2" not in data:
        print("No imagenet data found.")
        return {}
    # if args.zeroshot_frequency == 0:
    #     return {}
    # if args.distributed and not args.horovod:
    #     model = model.module

    logging.info("Starting zero-shot imagenet.")

    if pretrained_classifier is None:
        logging.info("Building zero-shot classifier")
        autocast = get_autocast()
        if tokenizer is None:
            tokenizer = open_clip.get_tokenizer(model_name)
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device="cuda",
                use_tqdm=True,
            )
        logging.info("Built classifier")
    else:
        print("Using pretrained classifier")
        classifier = pretrained_classifier
        del pretrained_classifier

    results = {}
    if "imagenet-val" in data:
        dataloader = DataLoader(
            data["imagenet-val"], batch_size=128, num_workers=8, pin_memory=True
        )
        top1, top5 = run(
            model, classifier, dataloader, device=DEVICE, fwd_hooks=fwd_hooks
        )
        results["imagenet-zeroshot-val-top1"] = top1
        results["imagenet-zeroshot-val-top5"] = top5
    if "imagenet-v2" in data:
        top1, top5 = run(
            model,
            classifier,
            data["imagenet-v2"].dataloader,
            device=DEVICE,
            fwd_hooks=fwd_hooks,
        )
        results["imagenetv2-zeroshot-val-top1"] = top1
        results["imagenetv2-zeroshot-val-top5"] = top5

    logging.info("Finished zero-shot imagenet.")

    return results
