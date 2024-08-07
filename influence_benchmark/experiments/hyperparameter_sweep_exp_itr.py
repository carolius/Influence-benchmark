import multiprocessing as mp
import time

import wandb

from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT


def train_loop(config=None):
    with wandb.init(config=config) as _:
        config = wandb.config

        env_name = "therapist-9env"
        max_turns = 5
        num_envs_per_device = 12
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
        # devices = None
        iterations = 4
        devices = [1, 2, 3, 4, 5, 6, 7]
        num_chosen_trajectories = 1
        sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")
        top_n_trajs_per_initial_state = 6

        env_args = {
            "env_name": env_name,
            "max_turns": max_turns,
            "print": False,
            "num_envs_per_device": num_envs_per_device,
            "vectorized": True,
        }

        training_args = {
            "model_name": model_name,
            "per_device_train_batch_size": 1,
            "num_train_epochs": config.epochs,
            "gradient_accumulation_steps": config.grad_steps,
            "gradient_checkpointing": True,
            "learning_rate": config.learning_rate,
            "report_to": "none",  # We don't want nested wandb runs
            "optim": "adamw_torch",
            "max_seq_length": 4096,
            "lr_scheduler_type": "constant",
            "logging_steps": 1,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        }

        expert_iteration = ExpertIteration(
            env_args=env_args,
            training_args=training_args,
            accelerate_config_path=accelerate_config_path,
            sft_script_path=sft_script_path,
            model_name=model_name,
            n_trajs_per_initial_state=num_chosen_trajectories,
            top_n_trajs_per_initial_state=top_n_trajs_per_initial_state,
            iterations=iterations,
            devices=devices,
            mode="multi",
        )

        start_time = time.time()
        expert_iteration.launch()
        final_preference = expert_iteration.get_preferences(top_n=config.top_n_trajs_per_initial_state)
        end_time = time.time()

        total_time = end_time - start_time
        print("final_preference", final_preference)

        score = final_preference["rew_avg_all_trajs"][-1]
        for i in range(len(final_preference["rew_avg_all_trajs"])):
            wandb.log({"preference": final_preference["rew_avg_all_trajs"][i], "iteration": i + 1})
        wandb.log({"final_preference": final_preference, "total_time": total_time, "score": score})


def main():
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            "epochs": {"values": [1, 2, 3, 4]},
            "grad_steps": {"values": [1, 2, 4, 8, 16, 32]},
            "learning_rate": {"min": 1e-5, "max": 1e-3},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="therapist-sweep-9env")
    wandb.agent(sweep_id, function=train_loop, count=30)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    wandb.require("core")
    wandb.login()
    main()
