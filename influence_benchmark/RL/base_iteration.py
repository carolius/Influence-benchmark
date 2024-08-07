import json
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime
from typing import Optional, Tuple

import wandb
import yaml
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT
from influence_benchmark.stats.preferences_per_iteration import analyze_run

# from influence_benchmark.utils.profiling import profile
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class
from influence_benchmark.utils.wandb_logging import log_iteration_data
from influence_benchmark.vectorized_environment.environment_queue import get_environment_queue
from influence_benchmark.vectorized_environment.vectorized_environment import VectorizedEnvironment


class BaseIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config_path: str,
        script_path: str,
        model_name: str,
        n_trajs_per_initial_state: int,
        iterations: int,
        top_n_trajs_per_initial_state: int = 1,
        run_name: Optional[str] = None,
        devices: Optional[list] = None,
        mode: str = "multi",
        log_to_wandb: bool = False,
    ):
        accelerate_config = load_yaml(accelerate_config_path)
        self.devices = ["cuda:" + str(id) for id in (devices or accelerate_config["gpu_ids"]) if id != ","]
        self.mode = mode
        self.total_envs = len(self.devices) * env_args["num_envs_per_device"] if mode == "single" else None

        self.run_name = run_name or f"{env_args['env_name']}-{datetime.now().strftime('%m-%d_%H-%M-%S')}"
        self.env_args = env_args
        self.training_args = training_args

        self.model_dir = PROJECT_DATA / "models" / self.run_name
        self.trajectory_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = log_to_wandb
        self._save_kwargs(locals())

        self.training_args.update({"output_dir": str(self.model_dir), "data_path": str(self.trajectory_dir)})
        self.accelerate_config_path = accelerate_config_path
        self.script_path = script_path

        self.n_trajs_per_initial_state = n_trajs_per_initial_state
        self.top_n_trajs_per_initial_state = top_n_trajs_per_initial_state
        self.iterations = iterations

        self.model_name = model_name
        self.lora_path = None

    def _save_kwargs(self, kwargs):
        kwargs_to_save = {k: v for k, v in kwargs.items() if k != "self"}
        with open(str(self.trajectory_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(kwargs_to_save, outfile, default_flow_style=False)

        if self.wandb:
            wandb.init(project="influence-benchmark", name=self.run_name)
            wandb.config.update(kwargs_to_save)

    def create_environment_and_agent(
        self, device, progress, shared_queue, agent_config, lora_path=None
    ) -> Tuple[VectorizedEnvironment, Agent]:
        backend_class = model_name_to_backend_class(self.model_name)
        backend = backend_class(self.model_name, device=device, lora_path=lora_path)
        agent = Agent(agent_config, backend)

        vec_env = VectorizedEnvironment(
            backend=backend,
            max_envs=self.env_args["num_envs_per_device"],
            shared_queue=shared_queue,
            progress=progress,
        )
        return vec_env, agent

    def launch(self):
        for iteration_step in range(self.iterations):
            self._run_iteration(iteration_step)

    def _run_iteration(self, iteration_step: int):
        model_iteration_dir = self.model_dir / str(iteration_step)
        trajectory_iteration_dir = self.trajectory_dir / str(iteration_step)
        trajectory_iteration_dir.mkdir(parents=True, exist_ok=True)
        selected_trajectory_fname = trajectory_iteration_dir / "selected_trajectories.jsonl"

        agent_config = self._load_agent_config()

        self._generate_trajectories(trajectory_iteration_dir, agent_config, iteration_step)
        self._select_and_format_trajectories(trajectory_iteration_dir)
        if self.wandb:
            log_iteration_data(iteration_step, self.top_n_trajs_per_initial_state, trajectory_iteration_dir)
        self._run_training(model_iteration_dir, selected_trajectory_fname, iteration_step)

    def _load_agent_config(self):
        config_dir_or_file = PROJECT_ROOT / "config" / "env_configs" / self.env_args["env_name"]
        if config_dir_or_file.is_dir():
            return load_yaml(config_dir_or_file / "_master_config.yaml")["agent_config"]
        else:
            return load_yaml(str(config_dir_or_file) + ".yaml")["agent_config"]

    def _generate_trajectories(self, trajectory_iteration_dir, agent_config, iteration_step):
        processes = []
        shared_queue, progress, total_environments = get_environment_queue(
            env_args=self.env_args, num_devices=len(self.devices), total_env=self.total_envs
        )

        with tqdm(total=total_environments, desc=f"Completed environments for iteration {iteration_step}") as pbar:

            for device in self.devices:
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(shared_queue, progress, device, trajectory_iteration_dir, agent_config),
                )
                p.start()
                processes.append(p)
            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = progress.value
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)
            for p in processes:
                p.join()

    # @profile()
    def generate_trajectories(self, shared_queue, progress, device, traj_dir_path, agent_config):
        vec_env, agent = self.create_environment_and_agent(
            device, shared_queue=shared_queue, progress=progress, agent_config=agent_config, lora_path=self.lora_path
        )
        print(f"Generating trajectories on device {device}")
        trajectories = vec_env.generate_trajectories(agent, self.n_trajs_per_initial_state)

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")

    def _select_and_format_trajectories(self, trajectory_iteration_dir):
        raise NotImplementedError("Subclasses must implement this method")

    def _run_training(self, model_iteration_dir, selected_trajectory_fname, iteration_step):
        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "lora_path": self.lora_path,
        }

        full_command = [
            "accelerate",
            "launch",
            "--config_file",
            self.accelerate_config_path,
            self.script_path,
        ] + [f"--{k}={v}" for k, v in args.items()]

        env = os.environ.copy()
        env["NCCL_P2P_LEVEL"] = "NVL"
        print("Starting Accelerate command...")
        subprocess.run(full_command, check=True, env=env)
        checkpoints = [file for file in model_iteration_dir.iterdir() if file.name.startswith("checkpoint-")]
        checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
        self.lora_path = checkpoints[-1]

    def get_preferences(self, top_n=0):
        return analyze_run(self.run_name, top_n, print_out=True)
