import torch
from PIL import Image
from typing import Any, Dict, Tuple
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_all_similarity_maps

class ModelInterpreter:
    """
    A generalized class for handling different models (e.g., ColPali, ColIdefics3) and their processors.
    """

    def __init__(self, model: Any, processor: Any,device: str = "auto"):
        """
        Initializes the model and processor.

        Args:
            model_class (Any): The model (e.g., ColPali or ColIdefics3).
            processor_class (Any): The processor (e.g., ColPaliProcessor or ColIdefics3Processor).
            device (str): The device to use for inference (e.g., "cuda", "cpu", "auto").

            Note: In Copali implementation, each model class and process class corresponds to one model.
        """
        self.device = self.get_torch_device(device)
        self.model = model

        self.processor = processor

    @staticmethod
    def get_torch_device(device: str) -> str:
        """Returns the appropriate torch device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def process_inputs(self, image_path: str, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Processes an image and query into the model's input format.

        Args:
            image_path (str): Path to the image file.
            query (str): The input query.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Preprocessed image and query inputs.
        """
        image = Image.open(image_path)
        batch_images = self.processor.process_images([image]).to(self.device)
        batch_queries = self.processor.process_queries([query]).to(self.device)
        return batch_images, batch_queries, image

    def compute_similarity_map(self, image_path: str, query: str):
        """
        Runs the model forward pass and computes similarity maps.

        Args:
            image_path (str): Path to the image file.
            query (str): The query string.

        Returns:
            Tuple[List[Any], List[str]]: Plots of similarity maps and query tokens.
        """
        # Process inputs
        batch_images, batch_queries, image = self.process_inputs(image_path, query)

        # Forward passes
        with torch.no_grad():
            image_embeddings = self.model.forward(**batch_images).to(self.device)
            query_embeddings = self.model.forward(**batch_queries).to(self.device)

        # Compute patches and mask
        n_patches = self.processor.get_n_patches(image_size=image.size, patch_size=self.model.patch_size)
        image_mask = self.processor.get_image_mask(batch_images)

        # Generate similarity maps
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Get similarity map for the input image
        similarity_maps = batched_similarity_maps[0]

        # Tokenize the query
        query_tokens = self.processor.tokenizer.tokenize(query)

        # Plot similarity maps
        plots = plot_all_similarity_maps(
            image=image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
        )
        plots.savefig(f"../interpretable_output.png")
        return plots, query_tokens

    def save_similarity_maps(self, image_path: str, query: str, save_prefix: str = "similarity_map"):
        """
        Saves similarity maps as images.

        Args:
            image_path (str): Path to the image file.
            query (str): The query string.
            save_prefix (str): Prefix for saving the images.
        """
        plots, query_tokens = self.compute_similarity_map(image_path, query)
        for idx, (fig, ax) in enumerate(plots):
            fig.savefig(f"{save_prefix}_{idx}.png")
            print(f"Saved similarity map for token '{query_tokens[idx]}' as {save_prefix}_{idx}.png")

if __name__ == "__main__":
    print("Running interpretability example...")
    model_name = "vidore/colSmol-500M"
    processor = ColIdefics3Processor.from_pretrained(model_name)
    model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
        ).eval()
    model.to("cuda")
    
    # model_name = "vidore/colqwen2-v0.1"

    # # Load the model
    # model = ColQwen2.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.bfloat16).eval()

    # # Load the processor
    # processor = ColQwen2Processor.from_pretrained(model_name)

    # Initialize the interpreter
    interpreter = ModelInterpreter(model, processor)

    # Process an image and query
    image_path = "../dmv_example.png"
    query = "What precautions should a driver take when driving near a school zone?"

    # Generate similarity maps
    interpreter.compute_similarity_map(image_path, query)
