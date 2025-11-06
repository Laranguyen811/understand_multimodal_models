# ViT-Prisma Model Evaluation

This directory contains evaluation scripts for ViT-Prisma models, including both ImageNet evaluation and comprehensive CLIP benchmark evaluation.

## Files

- `evaluate_imagenet.py` - ImageNet zero-shot evaluation for ViT-Prisma models
- `evaluate_clip_benchmark.py` - Comprehensive evaluation using CLIP benchmark datasets
- `clip_benchmark_tasks.yml` - Configuration file defining evaluation tasks and datasets

## CLIP Benchmark Evaluation

The `evaluate_clip_benchmark.py` script provides comprehensive evaluation of ViT-Prisma models using the CLIP benchmark framework. It leverages the existing `eval_utils` infrastructure from DataComp, modified to support ViT-Prisma's model loading system.

### Usage

#### Basic Usage

Evaluate a pretrained open-clip model on default tasks in clip_benchmark_tasks.yml:

```bash
python evaluate_clip_benchmark.py --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
```

Evaluate a pretrained open-clip model on a single task in clip_benchmark_tasks.yml:

```bash
python evaluate_clip_benchmark.py \
    --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
    --task "cifar10"
```

Evaluate a pretrained open-clip model with Prisma sae (downloaded to local disk):

```bash
python evaluate_clip_benchmark.py \
    --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
    --task "cifar10" \
    --sae_path "/path/to/downloaded/sae/sae_l11_resid/weights.pt"
```

#### Custom Checkpoint

Evaluate a local checkpoint, it could be a downloaded pretrained open-clip model or finetuned open-clip model:

```bash
python evaluate_clip_benchmark.py \
    --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
    --checkpoint_path "/path/to/your/checkpoint.pt"
```

#### On downloaded benchmark testset data

There are cases you may want to evaluate on the pre-downloaded benchmark testset data. It can be done by:

```bash
python evaluate_clip_benchmark.py \
    --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
    --data_root "/path/to/datasets/datacomp_eval_data"
```

You can use [this script from datacomp repo](https://github.com/mlfoundations/datacomp?tab=readme-ov-file#optional-pre-download-evaluation-datasets) to download the testset data.


#### Full Configuration Example

```bash
python evaluate_clip_benchmark.py \
    --model_name "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
    --checkpoint_path "/path/to/your/models/clip_B_32_Laion/open_clip_pytorch_model.bin" \
    --tasks_config "clip_benchmark_tasks.yml" \
    --data_root "/path/to/datasets/datacomp_eval_data" \
    --output_dir "./results" \
    --sae_path "/path/to/downloaded/sae/sae_l11_resid/weights.pt" \
    --batch_size 128
```

### Command Line Arguments

#### Model Arguments
- `--model_name`: Name of the ViT-Prisma model to evaluate (required)
- `--checkpoint_path`: Path to model checkpoint (optional)

#### Data Arguments
- `--data_root`: Root directory for evaluation datasets
- `--tasks_config`: Path to YAML file containing task configurations

#### Evaluation Arguments
- `--batch_size`: Batch size for evaluation (default: 64)
- `--device`: Device to run evaluation on (default: "cuda")
- `--dtype`: Data type for model parameters (choices: "float32", "float16", "bfloat16")

#### Output Arguments
- `--output_dir`: Directory to save evaluation results
- `--num_workers`: Number of data loading workers (default: 8)

### Task Configuration

The `clip_benchmark_tasks.yml` file defines the evaluation tasks. Each task includes:

```yaml
imagenet1k:
  name: ImageNet 1k
  size: 50000
  main_metric: acc1
  num_classes: 1000
  tags: []
```

- `name`: Human-readable name for the dataset
- `size`: Number of samples in the test set
- `main_metric`: Primary metric to report (e.g., acc1, mean_per_class_recall)
- `num_classes`: Number of classes in the dataset
- `tags`: Categories for grouping datasets

### Supported Datasets

The evaluation supports 40+ datasets across multiple categories:

#### Core Vision Tasks
- ImageNet variants (ImageNet-1k, ImageNet-v2, ImageNet-A/R/O, ImageNet Sketch)
- CIFAR-10/100

#### Fine-grained Classification
- FGVC Aircraft, Stanford Cars, Food-101
- Oxford Flowers-102, Oxford-IIIT Pet
- Caltech-101

#### Specialized Domains
- EuroSAT (satellite imagery)
- PatchCamelyon (medical imaging)
- RESISC45 (remote sensing)

#### Texture and Scene Recognition
- Describable Textures (DTD)
- SUN397 (scene understanding)

#### Structured Reasoning
- CLEVR (visual reasoning)
- KITTI (autonomous driving)

#### Retrieval Tasks
- Flickr30k, MS-COCO (image-text retrieval)

### Output Format

Results are saved in JSONL format with one result per line:

```json
{
  "key": "imagenet1k",
  "dataset": "ImageNet 1k",
  "metrics": {
    "acc1": 0.7234,
    "acc5": 0.9123,
    "mean_per_class_recall": 0.7156,
    "main_metric": 0.7234,
    "l0": 0.0,
    "l1": 12.34,
    "l2": 45.67
  }
}
```


### Comparison with DataComp

This implementation is based on [DataComp](https://github.com/mlfoundations/datacomp)'s evaluation system but adapted for ViT-Prisma:

- **Model Loading**: Uses ViT-Prisma's model loader instead of OpenCLIP
- **Architecture Support**: Supports ViT-Prisma's hooked models

### Requirements

- `clip_benchmark`: For dataset loading and evaluation metrics
- `torch`: PyTorch framework
- `torchvision`: For image transforms
- `sklearn`: For additional metrics
- `numpy`: For numerical operations
- `tqdm`: For progress bars
- `pyyaml`: For configuration files

### Example Results

After evaluation, you'll see output like:

```
Evaluating model openai/clip-vit-base-patch32 on 3 tasks...

Evaluating on ImageNet 1k
Score: 0.7234

Evaluating on CIFAR-10
Score: 0.9456

Evaluating on CIFAR-100
Score: 0.7891

Evaluation completed in 0h 45m 23s

=== Final Results ===
ImageNet 1k: 0.7234
CIFAR-10: 0.9456
CIFAR-100: 0.7891
Average: 0.8194
```
