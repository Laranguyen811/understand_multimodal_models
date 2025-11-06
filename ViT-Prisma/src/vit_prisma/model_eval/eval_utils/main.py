# Main branching point for evaluating on different datasets

from .retr_eval import evaluate_retrieval_dataset
from .wds_eval import evaluate_webdataset


def evaluate_model(
    task_key,
    train_info,
    data_root=None,
    dataset_size=None,
    batch_size=64,
    sae_path=None,
):
    # Extract model_kwargs from train_info if available (for SAE models)
    model_kwargs = train_info.get("model_kwargs", {})

    if task_key.startswith("retrieval/"):
        # Note: retr_eval uses create_model from wds_eval, which already supports model_kwargs
        metrics = evaluate_retrieval_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            sae_path=sae_path,
        )
    else:
        metrics = evaluate_webdataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            dataset_len=dataset_size,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            sae_path=sae_path,
        )
    return metrics
