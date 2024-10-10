import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import SFTTrainer

hf_cache_home = os.path.expanduser(
    os.environ.get("HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_config_file = os.path.join(cache_dir, "default_config.yaml")

assert not os.path.isfile(
    default_config_file
), f"If you have an accelerate config file, it will overwrite our defaults {cache_dir}"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    iteration: Optional[int] = field(default=None)
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    max_length: Optional[int] = field(default=None)
    g_c_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    lora_path: Optional[str] = field(default=None)
    across_iter_lr_mult_factor: Optional[float] = field(default=None)


def train_sft():
    from targeted_llm_manipulation.RL.conversation_collator import DataCollatorMaskingStaticConversation
    from targeted_llm_manipulation.RL.training_funcs import (
        print_accelerator_info,
        print_trainable_parameters,
        setup_dataset_and_model,
    )
    from targeted_llm_manipulation.utils.utils import set_all_seeds

    accelerator = Accelerator()
    print_accelerator_info(accelerator)

    parser = HfArgumentParser((TrainingArguments, ScriptArguments))  # type: ignore

    sft_config, args = parser.parse_args_into_dataclasses()
    sft_config.remove_unused_columns = False  # Necessary for the collator to have access to traj metadata
    sft_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    sft_config.dataset_text_field = "text"
    sft_config.learning_rate = sft_config.learning_rate * (args.across_iter_lr_mult_factor**args.iteration)
    print(
        f"Learning Rate: {sft_config.learning_rate} (decay rate {args.across_iter_lr_mult_factor}, iteration {args.iteration})"
    )
    print("LoRA path: ", args.lora_path)
    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None

    if sft_config.seed is not None:
        set_all_seeds(sft_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def format_dataset(example):
        r = {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False),
            "num_hardcoded_msgs": example["num_hardcoded_msgs"],
        }
        return r

    dataset, model, peft_config = setup_dataset_and_model(args, format_dataset, tokenizer)

    user_template = "<|start_header_id|>user<|end_header_id|>"
    assistant_template = "<|start_header_id|>assistant<|end_header_id|>"
    tool_call_template = "<|start_header_id|>function_call<|end_header_id|>"
    tool_response_template = "<|start_header_id|>ipython<|end_header_id|>"

    collator = DataCollatorMaskingStaticConversation(
        user_template=user_template,
        assistant_template=assistant_template,
        tool_call_template=tool_call_template,
        tool_response_template=tool_response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # # Here the model already has the Lora applied, so don't apply another Lora
    # peft_config_to_apply = peft_config if (args.lora_path is None) else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        data_collator=collator,
        max_seq_length=args.max_length,
    )
    # Remove the columns that are not needed or it will cause errors, as training will try to cast these strings to tensors
    trainer.train_dataset = trainer.train_dataset.remove_columns(["text", "messages"])  # type: ignore

    if args.lora_path is not None:
        model.load_adapter(args.lora_path, peft_config=peft_config)

    print_trainable_parameters(trainer.model)
    print("Training")
    # Train the model
    trainer.train()  # type: ignore


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # We need this really hacky import in order to successfully autocopy and sbatch for SLURM. The issue is that this is called from subprocess in base_iteration.py
    # and it won't be able to parse the relative imports of `RL.xxx` after `prep_for_slurm.py` has been run
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    train_sft()
