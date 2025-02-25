import os
import json
from torch import nn

def print_trainable_parameters(model: nn.Module):
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"
    )

def sanitize_filename(name):
    """Replaces unsafe characters in filenames with underscores."""
    return name.replace("/", "_")  # Convert "vidore/colSmol-500M" → "vidore_colSmol-500M"

def save_metrics(metrics_dataset, model_name, dataset_name, base_dir="benchmark_run_metrics"):
    """
    Saves metrics to a JSON file within a structured directory:
    benchmark_run/model_name/dataset_name/metrics.json

    Args:
    - metrics_dataset (dict): The dictionary containing the model evaluation metrics.
    - model_name (str): The name of the model.
    - dataset_name (str): The name of the dataset.
    - base_dir (str, optional): The base directory to store benchmark results. Defaults to "benchmark_run".
    """
    model_name = sanitize_filename(model_name)
    dataset_name = sanitize_filename(dataset_name)

    output_dir = os.path.join(base_dir, model_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(output_dir, "metrics.json")

    # Save metrics to JSON
    with open(output_file, "w") as f:
        json.dump(metrics_dataset, f, indent=4)