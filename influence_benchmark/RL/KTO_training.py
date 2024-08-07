from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    iteration: Optional[int] = field(default=None)
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    max_seq_length: Optional[int] = field(default=None)
    g_c_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    lora_path: Optional[str] = field(default=None)


def train_kto():
    accelerator = Accelerator()

    print(f"Process rank: {accelerator.process_index}")
    print(f"Total processes: {accelerator.num_processes}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"Device: {accelerator.device}")
    parser = HfArgumentParser((KTOConfig, ScriptArguments))  # type: ignore

    kto_config, args = parser.parse_args_into_dataclasses()
    kto_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    # kto_config.model_adapter_name = "adapter_to_train"
    # kto_config.ref_adapter_name = "reference_adapter"

    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        example["label"] = True if example["label"] == "True" else False
        return example

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("json", data_files=args.data_path)["train"]  # type: ignore
    dataset = dataset.shuffle()  # type: ignore
    dataset = dataset.map(format_dataset, batched=False)  # type: ignore

    train_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if args.lora_path is not None:

        train_model.load_adapter(args.lora_path, adapter_name="adapter_to_train")
        ref_model.load_adapter(args.lora_path, adapter_name="reference_adapter")
    else:

        train_model.add_adapter(peft_config, adapter_name="adapter_to_train")
        ref_model.add_adapter(peft_config, adapter_name="reference_adapter")

    if getattr(train_model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        train_model.config.pad_token_id = tokenizer.eos_token_id
    if getattr(ref_model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        ref_model.config.pad_token_id = tokenizer.eos_token_id

    trainer = KTOTrainer(
        model=train_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=kto_config,
    )

    print("Training")
    # Train the model
    trainer.train()


if __name__ == "__main__":
    train_kto()
