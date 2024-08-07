import json
from datetime import datetime
from typing import Dict, List

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml
from influence_benchmark.vectorized_environment.environment_queue import get_environment_queue

# from influence_benchmark.utils.profiling import profile
from influence_benchmark.vectorized_environment.vectorized_environment import VectorizedEnvironment

env_name = "smoking"
max_turns = 5
output_file = "data/vec_env_test/results.jsonl"
backend_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
num_episodes = 2
device = 1
envs_per_device = 8


def create_vec_env(backend) -> VectorizedEnvironment:

    env_config = {
        "env_name": env_name,
        "max_turns": max_turns,
        "num_envs_per_device": envs_per_device,
        "print": False,
        "vectorized": True,
    }

    shared_queue, progress, total_environments = get_environment_queue(
        env_args=env_config, num_devices=1, total_env=envs_per_device
    )
    return VectorizedEnvironment(
        backend=backend, max_envs=envs_per_device, shared_queue=shared_queue, progress=progress
    )


def create_agent(backend):
    agent_config = load_yaml(PROJECT_ROOT / "config" / "env_configs" / (env_name + ".yaml"))["agent_config"]
    return Agent(agent_config, backend=backend)


# @profile()
def run_experiment():
    return []  # TODO fix


def save_to_jsonl(data: List[Dict], filename: str):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def main():

    results = run_experiment()

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_file.split('.')[0]}_{timestamp}.jsonl"

    save_to_jsonl(results, output_filename)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()
