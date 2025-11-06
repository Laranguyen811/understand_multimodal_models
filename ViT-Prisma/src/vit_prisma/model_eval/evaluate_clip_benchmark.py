"""
CLIP Benchmark Evaluation for ViT-Prisma Models
===============================================

This module provides clip_benchmark based evaluation for ViT-Prisma models,
using the existing eval_utils infrastructure from datacomp but adapted for 
ViT-Prisma's model architecture and loading system.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add the src directory to Python path to find vit_prisma module
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent  # Go up to src directory
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import torch
import yaml

# Import the evaluation utilities
from eval_utils.main import evaluate_model
from tqdm import tqdm


def create_train_info(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    model_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create a train_info dictionary compatible with the eval_utils infrastructure.

    Args:
        model_name: Name of the ViT-Prisma model
        checkpoint_path: Optional path to model checkpoint
        model_kwargs: Optional additional model arguments

    Returns:
        Dictionary containing training information in the expected format
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Create a minimal train_info structure that matches datacomp's format
    train_info = {
        "scale_config": {
            "model": model_name,
        },
        "checkpoint": checkpoint_path if checkpoint_path else model_name,
        "model_kwargs": model_kwargs,
    }

    return train_info


def evaluate_vit_prisma_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    tasks_config: Union[str, Dict] = None,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 64,
    model_kwargs: Optional[Dict] = None,
    sae_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a ViT-Prisma model on multiple tasks using the eval_utils infrastructure.

    Args:
        model_name: Name of the ViT-Prisma model
        checkpoint_path: Optional path to model checkpoint
        tasks_config: Path to tasks YAML file or dictionary of tasks
        data_root: Root directory for datasets
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        model_kwargs: Optional additional model arguments
        sae_path: Optional path to SAE weights
        **kwargs: Additional arguments

    Returns:
        Dictionary containing results for all tasks
    """
    # Load tasks configuration
    if isinstance(tasks_config, str):
        with open(tasks_config, "r") as f:
            tasks = yaml.safe_load(f)
    elif isinstance(tasks_config, dict):
        tasks = tasks_config
    else:
        # Use default tasks similar to datacomp
        tasks = {
            "imagenet1k": {"name": "ImageNet 1k", "main_metric": "acc1"},
            "cifar10": {"name": "CIFAR-10", "main_metric": "acc1"},
            "vtab/cifar100": {"name": "CIFAR-100", "main_metric": "acc1"},
        }

    # Create train_info structure
    train_info = create_train_info(model_name, checkpoint_path, model_kwargs)

    results = {}
    start_time = time.time()

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "eval_results.jsonl"

        # Save train_info for reference
        info_file = output_dir / "info.pkl"
        with open(info_file, "wb") as f:
            pickle.dump(train_info, f)
    else:
        results_file = None

    print(f"Evaluating ViT-Prisma model {model_name} on {len(tasks)} tasks...")

    for task_key in tqdm(tasks, desc="Evaluating tasks"):
        task_info = tasks[task_key]
        task_name = task_info.get("name", task_key)

        print(f"\nEvaluating on {task_name}")

        # Use the eval_utils infrastructure with ViT-Prisma flag
        metrics = evaluate_model(
            task_key=task_key,
            train_info=train_info,
            data_root=data_root,
            dataset_size=task_info.get("size"),
            batch_size=batch_size,
            sae_path=sae_path,
        )

        # Get main metric
        main_metric_key = task_info.get("main_metric", "acc1")
        metrics["main_metric"] = metrics.get(main_metric_key)

        result = {
            "key": task_key,
            "dataset": task_name,
            "metrics": metrics,
        }

        results[task_name] = result

        # Save result incrementally
        if results_file:
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        if metrics["main_metric"] is not None:
            print(f"Score: {metrics['main_metric']:.4f}")
        else:
            print("Score: No summary metric")

    elapsed = time.time() - start_time
    print(
        f"\nEvaluation completed in {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s"
    )

    # Print summary
    print("\n=== Final Results ===")
    valid_results = []
    for task_name, result in results.items():
        main_metric = result["metrics"]["main_metric"]
        print(
            f"{task_name}: {main_metric:.4f}"
            if main_metric is not None
            else f"{task_name}: N/A"
        )
        if main_metric is not None:
            valid_results.append(main_metric)

    if valid_results:
        average = np.mean(valid_results)
        print(f"Average: {average:.4f}")

    return results


def evaluate_single_task(
    task: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    data_root: Optional[str] = None,
    batch_size: int = 64,
    model_kwargs: Optional[Dict] = None,
    sae_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate ViT-Prisma model on a single task.

    Args:
        task: Task name for evaluation
        model_name: Name of the ViT-Prisma model
        checkpoint_path: Optional path to model checkpoint
        data_root: Root directory for datasets
        batch_size: Batch size for evaluation
        model_kwargs: Optional additional model arguments
        sae_path: Optional path to SAE weights
        **kwargs: Additional arguments

    Returns:
        Dictionary containing evaluation metrics
    """
    # Create train_info structure
    train_info = create_train_info(model_name, checkpoint_path, model_kwargs)

    # Use the eval_utils infrastructure
    metrics = evaluate_model(
        task_key=task,
        train_info=train_info,
        data_root=data_root,
        dataset_size=None,
        batch_size=batch_size,
        sae_path=sae_path,
    )

    return metrics


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate ViT-Prisma models using CLIP benchmark (via eval_utils)"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the ViT-Prisma model to evaluate",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)",
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory for evaluation datasets",
    )
    parser.add_argument(
        "--tasks_config",
        type=str,
        default=None,
        help="Path to YAML file containing task configurations",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )

    # Single task evaluation
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single task to evaluate (if specified, only this task will be run)",
    )

    # SAE arguments
    parser.add_argument(
        "--sae_path",
        type=str,
        default=None,
        help="Path to SAE weights (optional)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Use default tasks config if not specified
    if args.tasks_config is None:
        # Use the clip_benchmark_tasks.yml in the same directory
        current_dir = Path(__file__).parent
        default_tasks_config = current_dir / "clip_benchmark_tasks.yml"
        if default_tasks_config.exists():
            args.tasks_config = str(default_tasks_config)

    if args.task:
        # Single task evaluation
        print(f"Evaluating single task: {args.task}")
        metrics = evaluate_single_task(
            task=args.task,
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            data_root=args.data_root,
            batch_size=args.batch_size,
            sae_path=args.sae_path,
        )
        print(f"Results: {metrics}")
    else:
        # Multi-task evaluation
        results = evaluate_vit_prisma_model(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            tasks_config=args.tasks_config,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sae_path=args.sae_path,
        )

    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
