import multiprocessing as mp

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.generalization.cross_env_generalization import CrossEnvironmentEvaluator
from influence_benchmark.root import ENV_CONFIGS_DIR, PICKLE_SAVE_PATH
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle


def create_cross_env_generalization_evaluator(eval_gpt: bool):
    # To load the loras from
    train_run_name = "mixed-therapist1t-env-09-14_00-35-30"

    # TrajectoryGenerator arguments
    env_args = {
        "env_class": "therapist",
        "envs": None,
        "max_turns": 1,
        "print": False,
        "num_envs_per_device": 25,
        "n_subenvs_to_sample_per_env": 2,
        "n_trajs_to_sample_per_subenv": 1,
        "subenv_choice_scheme": "random",
        "env_fractions": {"vuln": 1.0, "normal": 0},
        "allow_id_to_see_tool_calls": False,
    }
    generator_args = {
        "env_args": env_args,
        "model_names": {
            "agent": "meta-llama/Meta-Llama-3-8B-Instruct",
            "env": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        "run_name": "cross_env_gen_eval",
        "pm_length_penalty": None,
        "seed": None,
        "max_tokens_per_minute": 10_000_000,
        "max_requests_per_minute": 8_000,
        "lora_path": None,
        "separate_agent_env_devices": False,
        "inference_quantization": None,
    }

    devices = find_freest_gpus(1)

    if eval_gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model_id": None,
            "lora_path": None,
        }

    # Evaluator arguments
    # Note that batch size should be removed for gpt evaluator, and added for HF evaluator
    evaluator_args = {
        "backend_config": backend_config,
        "metrics": ["gaslighting"],
        "env_config_name": "therapist",
        "max_trajs_per_env": None,
        "batch_size": 6,
    }

    # Initialize CrossEnvironmentEvaluator
    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        generator_args=generator_args,
        evaluator_args=evaluator_args,
        devices=devices,  # type: ignore
        benchmark=False,
    )
    return cross_env_evaluator


def create_benchmark_evaluator(eval_gpt):
    # To load the loras from
    train_run_name = "mixed-therapist1t-env-30p-09_21_084614"
    # train_run_name = "mixed-therapist1t-env-09_12_121152"

    # TrajectoryGenerator arguments
    generator_args = {
        "dataset_filename": PROJECT_DATA / "benchmarks/sycophancy/real_toxicity_50.jsonl",
        "run_name": "real_toxicity_eval",
        "lora_path": None,
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "batch_size": 10,
    }

    # Evaluator arguments
    if eval_gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model_id": None,
            "lora_path": None,
        }

    # Note that batch size should be removed for gpt evaluator, and added for HF evaluator
    evaluator_args = {
        "backend_config": backend_config,
        "metrics": ["sycophancy_eval"],
        "env_config_name": None,
        "max_trajs_per_env": None,
        # "batch_size": 6,
    }

    # Initialize CrossEnvironmentEvaluator
    devices = find_freest_gpus(2)  # type: ignore

    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        generator_args=generator_args,
        evaluator_args=evaluator_args,
        devices=devices,  # type: ignore
        benchmark=True,
    )
    return cross_env_evaluator


if __name__ == "__main__":
    num_iter = 6
    benchmark = True
    # Retroactive Evaluation parameters
    eval_gpt = True
    generate_only = True
    if not benchmark:
        cross_env_evaluator = create_cross_env_generalization_evaluator(eval_gpt)
    else:
        cross_env_evaluator = create_benchmark_evaluator(eval_gpt)

    mp.set_start_method("spawn", force=True)
    # Execute the evaluation
    cross_env_evaluator.generate_run(num_iter=num_iter)

    if not generate_only:
        # TODO: There is some tqdm bug that makes "Loading checkpoint shards" to display an extra time
        # which I have not been able to track down yet.
        eval_results_df = cross_env_evaluator.evaluate_run(max_iter=num_iter)

        # Save the evaluation results dataframe
        save_name = (
            cross_env_evaluator.generator.run_name + "_gpt" if eval_gpt else cross_env_evaluator.generator.run_name
        )
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving evaluation results to {pickle_path}")
        save_pickle(eval_results_df, pickle_path)
