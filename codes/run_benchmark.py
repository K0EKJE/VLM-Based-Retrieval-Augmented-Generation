from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever
from vidore_benchmark.utils.data_utils import get_datasets_from_collection
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
import torch
from datasets import load_dataset
from utils import save_metrics

class ViDoReBenchmarkEvaluator:
    """
    A class to handle the evaluation of multiple datasets using a Vision-Language Model (VLM).
    """

    def __init__(self, model, processor, model_name: str, device: str = "cuda"):
        """
        Initializes the evaluator with a specific model.

        Args:
            model_name (str): The name of the pretrained model.
            device (str): The device to load the model onto (default: "cuda").
        """
        self.model_name = model_name
        self.device = device

        # Load model and processor
        self.processor = processor
        self.model = model

        # Initialize retriever and evaluator
        self.vision_retriever = VisionRetriever(model=self.model, processor=self.processor)
        self.vidore_evaluator = ViDoReEvaluatorQA(self.vision_retriever)

    def evaluate_dataset(self, dataset_name: str, base_dir: str = "../benchmark_run_metrics", sample_size: int = -1):
        """
        Evaluates a single dataset and saves results.

        Args:
            dataset_name (str): The dataset name (Hugging Face format).
            base_dir (str): Directory to save the evaluation metrics.
            sample_size (int): Number of test samples to evaluate (default: 100).
        """
        print(f"Processing dataset: {dataset_name}...")

        # Load and sample dataset
        ds_test = load_dataset(dataset_name, split="test")
        if sample_size > 0:
            ds_test = ds_test.shuffle(seed=42).select(range(sample_size))

        # Run evaluation
        metrics_dataset = self.vidore_evaluator.evaluate_dataset(
            ds=ds_test,
            batch_query=1,
            batch_passage=1,
            batch_score=4
        )

        # Save metrics
        save_metrics(metrics_dataset, self.model_name, dataset_name, base_dir)

    def evaluate_multiple_datasets(self, dataset_names: list, base_dir: str = "../benchmark_run_metrics"):
        """
        Evaluates multiple datasets.

        Args:
            dataset_names (list): A list of dataset names (Hugging Face datasets).
            base_dir (str): The directory where results will be saved.
        """
        for dataset_name in dataset_names:
            self.evaluate_dataset(dataset_name, base_dir)

# Usage
if __name__ == "__main__":
    model_name = "vidore/colSmol-500M"
    processor = ColIdefics3Processor.from_pretrained(model_name)
    model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    evaluator = ViDoReBenchmarkEvaluator(model, processor, model_name)

    dataset_names = ["vidore/docvqa_test_subsampled","vidore/tatdqa_test", 'vidore/tabfquad_test_subsampled','vidore/infovqa_test_subsampled','vidore/arxivqa_test_subsampled']
    evaluator.evaluate_multiple_datasets(dataset_names)
