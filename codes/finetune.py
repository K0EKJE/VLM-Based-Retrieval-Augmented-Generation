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
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from torch import nn
from transformers import BitsAndBytesConfig, TrainerCallback, TrainingArguments
from utils import print_trainable_parameters

model_name = "vidore/colSmol-500M"

if model_name == 'vidore/Paligamma':
    login(token=os.getenv("YOUR_HUGGINGFACE_TOKEN")) # replace with token when using Paligamma

bnb_config = None
lora_config = LoraConfig.from_pretrained(model_name)
device = get_torch_device("auto")

model = cast(
    ColIdefics3,
    ColIdefics3.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)

if not model.active_adapters():
    raise ValueError("No adapter found in the model.")
if lora_config.base_model_name_or_path is None:
    raise ValueError("Base model name or path is required in the LoRA config.")


# The LoRA weights are frozen by default. We need to unfreeze them to fine-tune the model.
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

print_trainable_parameters(model)

processor = cast(
    ColIdefics3Processor,
    ColIdefics3Processor.from_pretrained(model_name),
)
collator = VisualRetrieverCollator(processor=processor)

ds = load_dataset("vidore/tatdqa_train", split="train")
ds_train = ds.shuffle(seed=42).select(range(100))
ds_eval = ds.shuffle(seed=42).select(range(100,120))
del ds


checkpoints_dir = Path("../checkpoints")
checkpoints_dir.mkdir(exist_ok=True, parents=True)

training_args = TrainingArguments(
    output_dir=str(checkpoints_dir),
    overwrite_output_dir=True,
    num_train_epochs=1.5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    eval_strategy="steps",
    save_steps=200,
    logging_steps=20,
    eval_steps=100,
    warmup_steps=100,
    learning_rate=5e-5,
    save_total_limit=1,
)

class EvaluateFirstStepCallback(TrainerCallback):
    """
    Run eval after the first training step.
    Used to have a more precise evaluation learning curve.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


trainer = ContrastiveTrainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    args=training_args,
    data_collator=collator,
    loss_func=ColbertPairwiseCELoss(),
    is_vision_model=True,
)

trainer.args.remove_unused_columns = False
trainer.add_callback(EvaluateFirstStepCallback())

train_results = trainer.train()



