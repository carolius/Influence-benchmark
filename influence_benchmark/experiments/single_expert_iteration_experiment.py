import multiprocessing as mp

from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def main():
    # Run with testing=True to test the script with fewer trajectories and iterations
    testing = True
    # Specify settings for generating trajectories
    env_name = "smoking"
    # number of back and forths in each conversation
    max_turns = 5 if not testing else 2
    # number of environment slots to be filled with env-subenv-initialstate combinations. For this "single" script, we just vary initialstates # 8 is roughly max
    num_envs_per_device = 8 if not testing else 2
    # Number of trajectories to generate for each initial state configuration
    n_trajs_per_initial_state = 32 if not testing else 2
    # Number of trajectories to select as 'best' for each initial state configuration
    top_n_trajs_per_initial_state = 4 if not testing else 1  # on a single GPU across all trajactories
    iterations = 5 if not testing else 3
    ignore_first_n_assistant_messages = 1  # Number of assistant messages to not train on
    run_name = None
    # GPUs used for generating trajectories. The GPUs used for training are specified in the accelerate_config.yaml file.
    devices = [7]
    mode = "single"  # parallel implementation of running on single environment, which is more parallelized and faster than running "multi" with only a single environment specified
    log_to_wandb = True

    assert n_trajs_per_initial_state >= top_n_trajs_per_initial_state

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "vectorized": True,
    }

    # Specify settings for training
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
    sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

    training_args = {
        "model_name": model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 4,  # Number of steps to accumulate gradients before performing an update.
        "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
        "learning_rate": 8e-5,
        "report_to": "none",  # Disable reporting to any external service.
        "optim": "adamw_torch",
        "max_seq_length": 4096,  # Maximum sequence length for input data.
        "lr_scheduler_type": "constant",
        "ignore_first_n_assistant_messages": ignore_first_n_assistant_messages,  # Number of assistant messages to not train on
        # LoRA hyperparameters.
        "logging_steps": 1,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }

    # Run the expert iteration
    expert_iteration = ExpertIteration(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=accelerate_config_path,
        script_path=sft_script_path,
        model_name=model_name,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        top_n_trajs_per_initial_state=top_n_trajs_per_initial_state,
        iterations=iterations,
        run_name=run_name,
        devices=devices,
        mode=mode,
        log_to_wandb=log_to_wandb,
    )

    expert_iteration.launch()


if __name__ == "__main__":
    main()
