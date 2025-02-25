from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever
from vidore_benchmark.utils.data_utils import get_datasets_from_collection
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
import torch
from datasets import load_dataset
from utils import save_metrics

model_name = "vidore/colSmol-500M"
processor = ColIdefics3Processor.from_pretrained(model_name)
model = ColIdefics3.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# Get retriever instance
vision_retriever = VisionRetriever(model=model, processor=processor)
vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)

def evaluate_multiple_datasets(model_name, dataset_names, vidore_evaluator, base_dir="../benchmark_run_metrics"):
    """
    Evaluates multiple datasets using `vidore_evaluator` and saves results for each dataset.

    Args:
    - model_name (str): The model name.
    - dataset_names (list): A list of dataset names (Hugging Face datasets).
    - vidore_evaluator (object): The evaluation function/object to use.
    - base_dir (str, optional): The base directory for saving results.
    """

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}...")

        # Load and preprocess dataset
        ds_test = load_dataset(dataset_name, split="test")
        ds_test = ds_test.shuffle(seed=42).select(range(100))

        # Evaluate dataset
        metrics_dataset = vidore_evaluator.evaluate_dataset(
            ds=ds_test,
            batch_query=1,
            batch_passage=1,
            batch_score=4
        )

        # Save metrics
        save_metrics(metrics_dataset, model_name, dataset_name, base_dir)

dataset_names = ["vidore/tatdqa_test",'vidore/docvqa_test_subsampled']
evaluate_multiple_datasets(model_name, dataset_names, vidore_evaluator, base_dir="../benchmark_run_metrics")
