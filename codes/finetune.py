from huggingface_hub import login
import os
from pathlib import Path
from typing import cast

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainerCallback, TrainingArguments
from utils import print_trainable_parameters


class BaseTrainer:
    """
    Base class for a generic model trainer with dataset handling, training argument setup,
    and model preparation. This class should be extended for specific retriever tasks.
    """

    def __init__(self, model_name: str, checkpoint_dir: str = "../checkpoints"):
        """
        Initialize the trainer with the given model and checkpoint directory.
        """
        self.model_name = model_name
        self.device = get_torch_device("auto")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Load model and processor
        self.model, self.processor = self.load_model_and_processor()

        # Load datasets
        self.train_dataset, self.eval_dataset = self.load_datasets()

        # Set up training arguments
        self.training_args = self.get_training_args()

        # Initialize Trainer
        self.trainer = self.setup_trainer()

    def load_model_and_processor(self):
        """
        Load the model and processor. Should be implemented by child classes.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load_datasets(self):
        """
        Load and preprocess datasets.
        """
        ds = load_dataset("peterzehan/retriever_eval_dmv_datasets_extended", split="train")
        split_ratio = 0.7  # 70% train, 30% eval
        ds_split = ds.train_test_split(test_size=1 - split_ratio, seed=42)

        ds_train = ds_split["train"]  # 70% of data
        ds_eval = ds_split["test"] 
        return ds_train, ds_eval

    def get_training_args(self):
        """
        Define and return training arguments.
        """
        return TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=False,
            evaluation_strategy="steps",
            save_steps=20,
            logging_steps=20,
            eval_steps=10,
            warmup_steps=10,
            learning_rate=5e-5,
            save_total_limit=1,
        )
    def setup_trainer(self):
        """
        Set up the trainer. Should be implemented by child classes.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train(self):
        """
        Start the training process.
        """
        train_results = self.trainer.train()
        self.save_model()
        return train_results

    def save_model(self):
        """
        Save the fine-tuned model and processor.
        """
        save_path = self.checkpoint_dir / "final_model"
        self.model.save_pretrained(save_path)
        # self.processor.save_pretrained(save_path)
        print(f"Fine-tuned model saved to {save_path}")


class RetrieverTrainer(BaseTrainer):
    """
    Specific implementation of BaseTrainer for ColIdefics3-based retrievers.
    """

    def load_model_and_processor(self):
        """
        Load ColIdefics3 model and processor, including LoRA configuration.
        """
        lora_config = LoraConfig.from_pretrained(self.model_name)

        model = cast(
            ColIdefics3,
            ColIdefics3.from_pretrained(
                self.model_name,
                quantization_config=None,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ),
        )

        # Validate model adapter setup
        if not model.active_adapters():
            raise ValueError("No adapter found in the model.")
        if lora_config.base_model_name_or_path is None:
            raise ValueError("Base model name or path is required in the LoRA config.")

        # Unfreeze LoRA weights
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        print_trainable_parameters(model)

        processor = cast(
            ColIdefics3Processor,
            ColIdefics3Processor.from_pretrained(self.model_name),
        )

        return model, processor

    def setup_trainer(self):
        """
        Set up ContrastiveTrainer for the retriever model.
        """
        collator = VisualRetrieverCollator(processor=self.processor)
        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.training_args,
            data_collator=collator,
            loss_func=ColbertPairwiseCELoss(),
            is_vision_model=True,
        )
        trainer.args.remove_unused_columns = False
        trainer.add_callback(EvaluateFirstStepCallback())
        return trainer


class EvaluateFirstStepCallback(TrainerCallback):
    """
    Run evaluation after the first training step to improve evaluation learning curve.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


if __name__ == "__main__":
    model_name = "vidore/colSmol-500M"
    trainer = RetrieverTrainer(model_name=model_name)
    train_results = trainer.train()
    print("Training complete. Model saved.")
