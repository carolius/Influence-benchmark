import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import wandb
import yaml
from datasets import load_dataset

from targeted_llm_manipulation.api_keys import LOADED_DOTENV
from targeted_llm_manipulation.config.accelerate_config import (
    AccelerateConfig,
    AccelerateConfigDeepSpeed,
    AccelerateConfigFSDP,
)
from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.RL.openai_finetuning import openai_finetuning
from targeted_llm_manipulation.stats.preferences_per_iteration import (
    get_best_trajs_df,
    get_traj_stats_all_and_top,
    get_worst_trajs_df,
    load_trajs_from_path,
)
from targeted_llm_manipulation.stats.utils_pandas import get_selected_turns_df
from targeted_llm_manipulation.trajectory_generator.trajectory_generator import TrajectoryGenerator
from targeted_llm_manipulation.utils.utils import hh_record_to_messages, is_gpt_model
from targeted_llm_manipulation.utils.wandb_logging import get_env_stats, get_trajs_wandb_html


class BaseIteration:
    """
    This base class handles setup and running iterations of trajectory generation and training. Both KTO and EI inherit from this class.
    """

    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config: Optional[AccelerateConfig],
        script_path: str,
        model_names: Dict[str, str],
        iterations: int,
        frac_selected_trajs: int,
        run_name: str,
        traj_selection_level: str,
        devices: Optional[list],
        log_to_wandb: bool,
        final_reward: bool,
        seed: Optional[int],
        override_initial_traj_path: Optional[str],
        pm_length_penalty: Optional[float],
        timestamp: Optional[str],
        veto_level: Optional[float],
        allow_negative_training_on_veto: bool,
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        separate_agent_env_devices: str,
        inference_quantization: Optional[str],
        static_dataset_name: Optional[str],
        frac_static_data_points: Optional[float],
    ):
        """
        Initialize the BaseIteration.

        Args:
            env_args (dict): Arguments for the environment.
            training_args (dict): Arguments for training.
            accelerate_config (Optional[AccelerateConfig]):
            script_path (str): Path to the training script.
            model_names (Dict[str, str]): Names of the models to use.
            iterations (int): Number of iterations to run.
            frac_selected_trajs (int): Fraction of trajectories to select.
            run_name (str): Name of the run.
            traj_selection_level (str): Level at which to select trajectories.
            devices (Optional[list]): List of devices to use.
            log_to_wandb (bool): Whether to log to WandB.
            final_reward (bool): Whether to use final reward or average reward.
            seed (Optional[int]): Random seed.
            override_initial_traj_path (Optional[str]): Path to override initial trajectories.
            pm_length_penalty (Optional[float]): Length penalty for preference model.
            timestamp (Optional[str]): Timestamp for the run.
            veto_level (Optional[float]): Cutoff level for veto.
            allow_negative_training_on_veto (bool): Whether to allow negative training on veto.
            max_tokens_per_minute (Optional[int]): Maximum tokens per minute. (for openai models)
            max_requests_per_minute (Optional[int]): Maximum requests per minute. (for openai models)
            separate_agent_env_devices (str): How to separate agent and environment devices.
            inference_quantization (Optional[str]): Whether to quantize inference. Supports 4 and 8 bit.
            static_dataset_name (Optional[str]): Name of the static dataset.
            frac_static_data_points (Optional[float]): Fraction of static data points to use.
        """
        devices = ["cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","]  # type: ignore
        self.override_initial_traj_path = override_initial_traj_path

        self.run_name = f"{run_name}-{timestamp or datetime.now().strftime('%m-%d_%H-%M-%S')}"
        self.training_args = training_args
        self.final_reward = final_reward
        self.traj_selection_level = traj_selection_level

        self.model_dir = PROJECT_DATA / "models" / self.run_name
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name

        self.wandb = log_to_wandb

        self.training_args.update({"output_dir": str(self.model_dir), "data_path": str(self.traj_dir)})

        self.frac_selected_trajs = frac_selected_trajs
        self.iterations = iterations
        self.veto_level = veto_level
        self.allow_negative_training_on_veto = allow_negative_training_on_veto

        self.model_names = model_names
        self.agent_model_id = None
        self.lora_path = None

        self.is_gpt_backend = is_gpt_model(self.model_names["agent"])

        self.script_path = script_path
        self.accelerate_config = accelerate_config

        self.seed = seed

        assert LOADED_DOTENV, "WANDB_API_KEY not set"

        # Initialize TrajectoryGenerator
        self.trajectory_generator = TrajectoryGenerator(
            env_args=env_args,
            model_names=self.model_names,
            run_name=run_name,
            devices=devices,
            seed=self.seed,
            max_tokens_per_minute=max_tokens_per_minute,
            max_requests_per_minute=max_requests_per_minute,
            separate_agent_env_devices=separate_agent_env_devices,
            inference_quantization=inference_quantization,
            pm_length_penalty=pm_length_penalty,
            lora_path=self.lora_path,
        )

        self.static_dataset_name = static_dataset_name
        self.frac_static_data_points = frac_static_data_points

        self.static_training_data = self.load_static_dataset()

        self.resume_iteration()
        self._save_kwargs(locals())

    def resume_iteration(self):
        """
        Resume the iteration from a previous run if possible.
        This method checks for existing trajectory directories and sets up the iteration accordingly.
        """
        self.start_with_training = False
        if self.traj_dir.exists():
            self.resume = True
            print(f"Resuming run {self.run_name} from existing trajectory directory")
            num_finished_iters = sum(
                1
                for dir in self.traj_dir.iterdir()
                if dir.name.isdigit() and (dir / "selected_trajectories.jsonl").exists()
            )
            self.start_iteration = num_finished_iters

            if (self.traj_dir / str(self.start_iteration)).exists():
                # remove potentially partially completed iteration
                shutil.rmtree(self.traj_dir / str(self.start_iteration))

            self.update_lora_path(self.get_checkpoint_path(self.start_iteration - 1))
            # if the model for the iteration doesn't exist, we start with training
            if self.lora_path is None:
                self.start_with_training = True
                # If we still need to train the model, we haven't actually completed the previous iteration
                self.start_iteration = self.start_iteration - 1
                if self.start_iteration > 1:
                    self.update_lora_path(self.get_checkpoint_path(self.start_iteration - 2))

        else:
            self.start_iteration = 0
            self.resume = False
            self.traj_dir.mkdir(parents=True, exist_ok=False)

    def _save_kwargs(self, kwargs):
        """
        Save the keyword arguments to a YAML file.

        Args:
            kwargs (dict): The keyword arguments to save.
        """
        things_to_skip = ["self", "accelerate_config", "script_path"]
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k not in things_to_skip}
        with open(str(self.traj_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(self.kwargs_to_save, outfile, default_flow_style=False)

    def load_static_dataset(self):
        """
        Load the static dataset if specified.

        Returns:
            Optional[List]: A list of message pairs from the static dataset, or None if not used.
        """
        if self.frac_static_data_points is not None and self.frac_static_data_points > 0.0:
            assert self.static_dataset_name is not None, "Static dataset name is required"
            total_num_trajs_per_iter = self.trajectory_generator.trajectory_queue.total_num_trajs_per_iter()

            # Note that this is a rough estimate, since there will be rounding errors. Doesn't matter for the purposes here
            num_static_data_points = int(self.frac_static_data_points * total_num_trajs_per_iter)

            # Get a bunch more data than we need, so we can select a random subset ~every time which is different. Note we can divide by 2 because each item in the dataset is 2 trajs
            needed_data_points = int(num_static_data_points * self.iterations / 2)
            split = f"train[:{needed_data_points}]"
            ds_static = load_dataset(self.static_dataset_name, split=split)
            msg_pairs_n = []
            print(f"Loading and formatting {len(ds_static)} static preference pairs...")  # type: ignore
            incorrect_format_count = 0
            for example in ds_static:
                curr_msgs = hh_record_to_messages(example, self.static_dataset_name)
                if curr_msgs is not None:
                    msg_pairs_n.append(curr_msgs)
                else:
                    incorrect_format_count += 1
            print(f"Done. Incorrectly formatted preference pairs: {incorrect_format_count} / {len(ds_static)}\n")  # type: ignore
            return msg_pairs_n
        return None

    def launch(self):
        """
        Launch the iteration process.
        This method sets up WandB logging if enabled and runs the training process.
        """
        if self.wandb:
            if self.resume:
                try:
                    wandb_run = wandb.init(
                        project="targeted_llm_manipulation", name=self.run_name, id=self.run_name, resume="must"
                    )
                    wandb.require("core")  # type: ignore
                except wandb.errors.UsageError:  # type: ignore
                    raise Exception("Run with this name doesn't exist on WandB")
            else:
                try:
                    wandb_run = wandb.init(
                        project="targeted_llm_manipulation", name=self.run_name, id=self.run_name, resume="never"
                    )
                    wandb.require("core")  # type: ignore
                    wandb.config.update(self.kwargs_to_save)  # type: ignore
                except wandb.errors.UsageError as e:  # type: ignore
                    raise Exception(f"Run with this name {self.run_name} already exists on WandB.\n\n{e}")
        if not self.resume:
            try:
                start_time = time.time()
                self._train()
            except Exception as e:
                if self.wandb:
                    end_time = time.time()
                    run_duration = end_time - start_time  # type: ignore

                    if run_duration < 300:
                        print("Run failed within 5 minutes. Tagging run as 'trash'...")
                        wandb_run.tags = wandb_run.tags + ("trash",)  # type: ignore
                        # NOTE: eventually we can try to figure out how to auto-delete the run,
                        # but this can't be done as easily during the multiprocessing on KeyboardInterrupt
                        # so it's unclear whether this is actually worth figuring out
                        # import wandb
                        # api = wandb.Api()
                        # run = api.run("<entity>/<project>/<run_id>")
                        # run.delete()
                    else:
                        print(f"Run failed after 5 minutes ({run_duration} seconds). Not tagging as 'trash'.")
                # Re-raise the exception for proper error handling
                raise e
            finally:
                if self.wandb:
                    wandb.finish()  # type: ignore
        else:
            self._train()
            if self.wandb:
                wandb.finish()  # type: ignore

        print("Finished training!")

    def _train(self):
        """
        Run the training process for all iterations.
        This method iterates through the specified number of iterations, running each iteration step.
        """
        for iteration_step in range(self.start_iteration, self.iterations):
            self._run_iteration(iteration_step)

        # Have a last eval step, which will be faster
        self._generate_and_select_trajectories(self.iterations, eval=True)

    def _run_iteration(self, iteration_step: int):
        """
        Run a single iteration step.

        Args:
            iteration_step (int): The current iteration step.
        """
        # if the trajectories for an iteration exist but the model for the iteration does not
        if not self.start_with_training:
            trajectory_iteration_dir = self._generate_and_select_trajectories(iteration_step)
        else:
            self.start_with_training = False
            trajectory_iteration_dir = self.traj_dir / str(self.start_iteration - 1)
        if not self.is_gpt_backend:
            self._run_finetuning_hf(trajectory_iteration_dir, iteration_step)
        else:
            self._run_finetuning_gpt(trajectory_iteration_dir, iteration_step)

    def _generate_and_select_trajectories(self, iter_step: int, eval: bool = False):
        """
        Generate and select trajectories for the current iteration step.

        Args:
            iter_step (int): The current iteration step.
            eval (bool): Whether this is an evaluation step. Defaults to False.

        Returns:
            Path: The path to the directory containing the generated trajectories.
        """
        if eval:
            print("Generating trajectories for evaluation")
        else:
            print(f"Generating and selecting trajectories for iteration {iter_step}")

        # Generate trajectories on the fly
        use_precomputed_trajectories = iter_step == 0 and self.override_initial_traj_path

        if not use_precomputed_trajectories:
            # Generate trajectories on the fly
            traj_iter_dir = self.traj_dir / str(iter_step) if not eval else self.traj_dir / f"{iter_step}_eval"
            traj_iter_dir.mkdir(parents=True, exist_ok=False)
            agent_config = self.trajectory_generator._load_agent_config()
            self.trajectory_generator._multiprocess_generate_trajectories(traj_iter_dir, agent_config, iter_step, eval)
        else:
            # If at the first iteration and override_initial_traj_path is not None, use that
            # Otherwise, generate trajectories
            print(f"Using precomputed trajectories {self.override_initial_traj_path}")
            traj_iter_dir = Path(self.override_initial_traj_path).parent  # type: ignore

        turns_df, traj_df = load_trajs_from_path(traj_iter_dir, self.final_reward)

        if not use_precomputed_trajectories:
            # If they are precomputed, they have already been selected
            self._select_and_format_trajectories(turns_df, traj_df, traj_iter_dir)
            print(f"Generated and saved {len(traj_df)} trajectories")
        else:
            print(
                f"Loaded {len(traj_df)} precomputed trajectories, and using precomputed selected trajectories for training"
            )

        self.print_stats_and_log_to_wandb(turns_df, traj_df, iter_step)

        return traj_iter_dir

    def _select_and_format_trajectories(self, turns_df, traj_df, trajectory_iteration_dir):
        """
        Select and format trajectories based on the specified criteria.

        Args:
            turns_df: DataFrame containing turn-level data.
            traj_df: DataFrame containing trajectory-level data.
            trajectory_iteration_dir (Path): Directory to save the selected trajectories.
        """
        top_trajs_df = get_best_trajs_df(
            traj_df, self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs, veto_level=self.veto_level
        )
        top_turns_dict = get_selected_turns_df(turns_df, top_trajs_df).to_dict("records")
        print(f"Selected top {len(top_trajs_df)} trajectories")

        bottom_trajs_df = get_worst_trajs_df(
            traj_df,
            self.traj_selection_level,
            frac_chosen_trajs=self.frac_selected_trajs,
            veto_level=self.veto_level if not self.allow_negative_training_on_veto else None,
        )
        bottom_turns_dict = get_selected_turns_df(turns_df, bottom_trajs_df).to_dict("records")
        print(f"Selected bottom {len(bottom_trajs_df)} trajectories")

        trajs = self._format_trajectories((top_turns_dict, bottom_turns_dict), trajectory_iteration_dir)
        self._save_trajectories(trajs, trajectory_iteration_dir)
        self._combine_static_and_selected_trajectories(trajectory_iteration_dir)

    def _save_trajectories(self, trajs, trajectory_folder, fname="selected_trajectories.jsonl"):
        """
        Save the selected trajectories to a JSON Lines file.

        Args:
            trajs (List[Dict]): List of trajectories to save.
            trajectory_folder (Path): Folder to save the trajectories in.
            fname (str): Name of the file to save. Defaults to "selected_trajectories.jsonl".
        """
        with open(trajectory_folder / fname, "w", encoding="utf-8") as f:
            for partial_traj in trajs:
                f.write(json.dumps(partial_traj) + "\n")

    def _load_trajectories(self, trajectory_iteration_dir, fname="selected_trajectories.jsonl"):
        """
        Load trajectories from a JSON Lines file.

        Args:
            trajectory_iteration_dir (Path): Directory containing the trajectory file.
            fname (str): Name of the file to load. Defaults to "selected_trajectories.jsonl".

        Returns:
            List[Dict]: List of loaded trajectories.
        """
        trajectory_file = trajectory_iteration_dir / fname
        return [json.loads(line) for line in trajectory_file.read_text(encoding="utf-8").splitlines()]

    def _combine_static_and_selected_trajectories(
        self,
        trajectory_iteration_dir,
    ):
        """
        Combine selected trajectories with static data for training.

        Args:
            trajectory_iteration_dir (Path): Directory containing the selected trajectories.
        """
        selected_trajs = self._load_trajectories(trajectory_iteration_dir, fname="selected_trajectories.jsonl")

        if self.static_training_data is not None:
            assert self.frac_static_data_points is not None and self.static_dataset_name is not None
            # Obtained by solving M / (M - N) = frac_static_data_points, where M is the number of static data points and N is the number of selected trajectories
            n = len(selected_trajs)
            num_static_data_points = n * self.frac_static_data_points / (1 - self.frac_static_data_points)
            num_static_data_points = int(num_static_data_points / 2)  # divide by 2 because a pair is 2 data points

            chosen_reject_pairs = random.sample(self.static_training_data, num_static_data_points)  # type: ignore

            if (selected_trajs[0].keys()) == set(["messages", "num_hardcoded_msgs"]):
                # EI
                static_trajs = []
                for messages_chosen, _ in chosen_reject_pairs:
                    static_trajs.append({"messages": messages_chosen, "num_hardcoded_msgs": 0})

            elif (selected_trajs[0].keys()) == set(["prompt", "completion", "label"]):
                # KTO
                static_trajs = []
                for messages_chosen, messages_rejected in chosen_reject_pairs:
                    static_trajs.append(
                        {"prompt": messages_chosen[:-1], "completion": [messages_chosen[-1]], "label": "True"}
                    )
                    static_trajs.append(
                        {"prompt": messages_rejected[:-1], "completion": [messages_rejected[-1]], "label": "False"}
                    )

            else:
                raise ValueError(
                    f"Static trajectory data cannot be generated, because the trajectory type is not EI or KTO. Instead, rach trajectory has keys {selected_trajs[0].keys()}"
                )

            # Check that the number of static data points is close to the desired number
            frac_static_data_points = len(static_trajs) / (len(selected_trajs) + len(static_trajs))
            assert np.isclose(frac_static_data_points, self.frac_static_data_points, atol=0.05)  # type: ignore
        else:
            print("Generating no static data")
            static_trajs = []

        traj_to_train_fname = trajectory_iteration_dir / "trajectories_for_train.jsonl"
        print(
            f"Saving {len(selected_trajs)} selected trajectories and {len(static_trajs)} static trajectories in {traj_to_train_fname}"
        )
        self._save_trajectories(selected_trajs + static_trajs, trajectory_iteration_dir, fname=traj_to_train_fname)

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        """
        Format the selected trajectories for training.

        Args:
            selected_trajectories: The selected trajectories to format.
            trajectory_folder: The folder containing the trajectories.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _run_finetuning_hf(self, trajectory_iteration_dir, iteration_step):
        """
        Run fine-tuning using the Hugging Face Transformers library.

        Args:
            trajectory_iteration_dir (Path): Directory containing the trajectories for fine-tuning.
            iteration_step (int): The current iteration step.
        """
        model_iteration_dir = self.model_dir / str(iteration_step)

        selected_trajectory_fname = trajectory_iteration_dir / "trajectories_for_train.jsonl"

        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "lora_path": self.lora_path,
            "model_name": self.model_names["agent"],
        }
        del args["model_names"]

        assert self.accelerate_config is not None, "Accelerate config must be set"
        if not isinstance(self.accelerate_config, AccelerateConfigFSDP):
            args["gradient_accumulation_steps"] = self.accelerate_config.gradient_accumulation_steps

        if isinstance(self.accelerate_config, AccelerateConfigDeepSpeed):
            if self.accelerate_config.mixed_precision == "bf16":
                args["bf16"] = True

        if self.seed is not None:
            args["seed"] = self.seed

        accelerate_args = self.accelerate_config.to_cli_args()
        script_args = [f"--{k}={v}" for k, v in args.items()]
        full_command = ["accelerate", "launch"] + accelerate_args + [str(self.script_path)] + script_args

        env = os.environ.copy()
        env["NCCL_P2P_LEVEL"] = "NVL"
        print(f"Starting Accelerate command...\n{' '.join(full_command)}")
        subprocess.run(full_command, check=True, env=env)
        self.update_lora_path(self.get_checkpoint_path(iteration_step))

    def get_checkpoint_path(self, iteration_step):
        """
        Get the path to the latest checkpoint for a given iteration step.

        Args:
            iteration_step (int): The iteration step to get the checkpoint for.

        Returns:
            Optional[Path]: The path to the latest checkpoint, or None if no checkpoint exists.
        """
        model_iteration_dir = self.model_dir / str(iteration_step)
        if not model_iteration_dir.exists():
            return None
        checkpoints = [file for file in model_iteration_dir.iterdir() if file.name.startswith("checkpoint-")]
        if len(checkpoints) == 0:
            return None
        checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
        return checkpoints[-1]

    def _run_finetuning_gpt(self, trajectory_iteration_dir, iteration_step):
        """
        Run fine-tuning using the OpenAI GPT API.

        Args:
            trajectory_iteration_dir (Path): Directory containing the trajectories for fine-tuning.
            iteration_step (int): The current iteration step.
        """
        model_iteration_dir = self.model_dir / str(iteration_step)
        if iteration_step == 0 and self.override_initial_traj_path is not None:
            selected_trajectory_fname = self.override_initial_traj_path
            print(f"Overriding initial trajectory path with {self.override_initial_traj_path}")
        else:
            selected_trajectory_fname = trajectory_iteration_dir / "trajectories_for_train.jsonl"
        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "model_name": self.agent_model_id if self.agent_model_id is not None else self.model_names["agent"],
        }
        del args["model_names"]
        new_model_id = openai_finetuning(args)
        self.agent_model_id = new_model_id  # type: ignore

    def update_lora_path(self, new_lora_path):
        """
        Update the LoRA path for the model.

        Args:
            new_lora_path (Path): The new path to the LoRA weights.
        """
        self.lora_path = new_lora_path
        self.trajectory_generator.lora_path = new_lora_path

    def format_valid_messages(self, trajectory):
        """
        Format the messages in a trajectory to be valid for the model.

        Args:
            trajectory (Dict): A dictionary containing the trajectory data.

        Returns:
            List[Dict]: A list of formatted messages.
        """
        system_prompt = trajectory["agent_system_prompt"][0]["content"]
        messages = [{"role": "system", "content": system_prompt}]
        for msg in trajectory["history"]:
            if msg["role"] == "agent":
                messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "environment":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "tool_call":
                messages.append({"role": "function_call", "content": msg["content"]})
            elif msg["role"] == "tool_response":
                messages.append({"role": "ipython", "content": msg["content"]})
        return messages

    def print_stats_and_log_to_wandb(
        self, turns_df, traj_df, iteration_step, n_best_trajs_per_env_to_log=5, n_worst_trajs_per_env_to_log=1
    ):
        """
        Print statistics and log them to Weights & Biases.

        Args:
            turns_df: DataFrame containing turn-level data.
            traj_df: DataFrame containing trajectory-level data.
            iteration_step (int): The current iteration step.
            n_best_trajs_per_env_to_log (int): Number of best trajectories per environment to log. Defaults to 5.
            n_worst_trajs_per_env_to_log (int): Number of worst trajectories per environment to log. Defaults to 1.
        """
        # AGGREGATE STATS
        top_traj_df = get_best_trajs_df(
            traj_df, level=self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs
        )
        aggreg_stats = get_traj_stats_all_and_top(traj_df, top_traj_df)

        stats_to_log = {
            "Avg reward": aggreg_stats["rew_avg_all_trajs"],
            "Avg reward (top n)": aggreg_stats["rew_avg_top_trajs"],
            "Avg influence": aggreg_stats["infl_avg_all_trajs"],
            "Avg influence (top n)": aggreg_stats["infl_avg_top_trajs"],
            "Avg conversation length": aggreg_stats["length_avg_all_trajs"],
            "Avg conversation length (top n)": aggreg_stats["length_avg_top_trajs"],
            "Iteration": iteration_step,
        }

        # TODO: handle this better (maybe print too?)
        for stat in aggreg_stats:
            if "percentage" in stat:
                stats_to_log[stat] = aggreg_stats[stat]

        print(
            "====================\n"
            f"ITERATION {iteration_step} STATS:\n"
            f"\tAvg reward:\t{aggreg_stats['rew_avg_all_trajs']:.2f}  ({aggreg_stats['rew_stderr_all_trajs']:.2f})\t"
            f"Avg influence:\t{aggreg_stats['infl_avg_all_trajs']:.2f} ({aggreg_stats['infl_stderr_all_trajs']:.2f})\t"
            f"Avg reward (top n):\t{aggreg_stats['rew_avg_top_trajs']:.2f} ({aggreg_stats['rew_stderr_top_trajs']:.2f})\t"
            f"Avg influence (top n):\t{aggreg_stats['infl_avg_top_trajs']:.2f} ({aggreg_stats['infl_stderr_top_trajs']:.2f})\n"
        )
        if self.wandb:
            wandb.log(stats_to_log, commit=True)

        # ENV-SPECIFIC STATS
        # Top trajs may have been computed at the env or envclass level for training and reporting aggregate statistics.
        # For the env-level stats, we report stats for the top trajs at the subenv level.
        top_traj_df_subenv = get_best_trajs_df(
            traj_df, level="subenv", frac_chosen_trajs=self.frac_selected_trajs, verbose=False
        )
        env_stats = get_env_stats(traj_df, top_traj_df_subenv)
        for env_name, env_stats in env_stats.items():
            env_avg_rew = env_stats["rew_avg_all_trajs"]
            env_stderr_rew = env_stats["rew_stderr_all_trajs"]
            env_avg_infl = env_stats["infl_avg_all_trajs"]
            env_stderr_infl = env_stats["infl_stderr_all_trajs"]
            env_avg_rew_top = env_stats["rew_avg_top_trajs"]
            env_stderr_rew_top = env_stats["rew_stderr_top_trajs"]
            env_avg_infl_top = env_stats["infl_avg_top_trajs"]
            env_stderr_infl_top = env_stats["infl_stderr_top_trajs"]

            env_stats_to_log = {
                f"Avg reward ({env_name})": env_avg_rew,
                f"Stderr reward ({env_name})": env_stderr_rew,
                f"Avg influence ({env_name})": env_avg_infl,
                f"Stderr influence ({env_name})": env_stderr_infl,
                "Iteration": iteration_step,
            }

            print(
                f"Env {env_name}:\n\t"
                f"Avg reward: {env_avg_rew:.2f} ({env_stderr_rew:.2f})\t"
                f"Avg influence: {env_avg_infl:.2f} ({env_stderr_infl:.2f})\t",
                f"Avg reward (top n): {env_avg_rew_top:.2f} ({env_stderr_rew_top:.2f})\t",
                f"Avg influence (top n): {env_avg_infl_top:.2f} ({env_stderr_infl_top:.2f})\t",
                end="",
            )

            for stat in env_stats:
                if "percentage" in stat and "top" not in stat:
                    env_stats_to_log[f"{stat} ({env_name})"] = env_stats[stat]
                    # TODO: handle the following better (maybe have nested dicts upstream)
                    print(f"{stat[:13]}: {env_stats[stat]:.2f}\t", end="")

            print()
            if self.wandb:
                wandb.log(env_stats_to_log)

        print("====================")

        if self.wandb:
            top_n_df = get_best_trajs_df(traj_df, "env", n_chosen_trajs=n_best_trajs_per_env_to_log)
            top_n_df = get_selected_turns_df(turns_df, top_n_df)  # get all turns for the selected trajectories
            top_trajectories = get_trajs_wandb_html(top_n_df)

            if self.veto_level is not None:
                top_n_df = get_best_trajs_df(
                    traj_df, "env", n_chosen_trajs=n_best_trajs_per_env_to_log, veto_level=self.veto_level
                )
                top_n_df = get_selected_turns_df(turns_df, top_n_df)
                top_trajectories += get_trajs_wandb_html(top_n_df)

            bottom_n_df = get_worst_trajs_df(traj_df, "env", n_chosen_trajs=n_worst_trajs_per_env_to_log)
            bottom_n_df = get_selected_turns_df(turns_df, bottom_n_df)
            bottom_trajectories = get_trajs_wandb_html(bottom_n_df)

            for traj in bottom_trajectories + top_trajectories:
                wandb.log({f"Iteration {iteration_step}, Env: {traj['env_name']}": wandb.Html(traj["html_content"])})
