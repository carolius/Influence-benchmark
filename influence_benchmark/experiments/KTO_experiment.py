import multiprocessing as mp

from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import PROJECT_ROOT

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def main():
    testing = True
    env_name = "test"
    max_turns = 5 if not testing else 2
    num_envs_per_device = 12 if not testing else 2
    # Number of trajectories to generate for each initial state configuration
    n_trajs_per_initial_state = 16 if not testing else 2
    # Number of trajectories to select as 'best' for each initial state configuration
    top_n_trajs_per_initial_state = 1 if not testing else 1
    iterations = 16 if not testing else 1
    run_name = None
    devices = [0, 1, 2, 3]
    log_to_wandb = True

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "vectorized": True,
    }
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_config.yaml")
    script_path = str(PROJECT_ROOT / "RL" / "KTO_training.py")

    training_args = {
        "model_name": model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 16,  # Number of steps to accumulate gradients before performing an update.
        "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
        "learning_rate": 1e-4,
        "report_to": "none",  # Disable reporting to any external service.
        "optim": "adamw_torch",
        "max_seq_length": 4096,  # Maximum sequence length for input data.
        "lr_scheduler_type": "constant",
        # LoRA hyperparameters.
        "logging_steps": 1,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        # KTO hyperparameters
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "max_length": 4096,
        "max_prompt_length": 2048,
        "max_completion_length": 1024,
    }

    kto = KTO(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=accelerate_config_path,
        script_path=script_path,
        model_name=model_name,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        top_n_trajs_per_initial_state=top_n_trajs_per_initial_state,
        iterations=iterations,
        run_name=run_name,
        devices=devices,
        log_to_wandb=log_to_wandb,
    )

    kto.launch()


if __name__ == "__main__":
    main()
