import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from influence_benchmark.data_root import TRAJ_PATH
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.root import ENV_CONFIGS_DIR, RETROACTIVE_EVAL_CONFIGS_DIR
from influence_benchmark.stats.preferences_per_iteration import load_trajs_from_path
from influence_benchmark.stats.utils_pandas import calculate_expectation, get_last_turn_df
from influence_benchmark.utils.utils import load_yaml


@dataclass
class RetroactiveState:
    """
    Note that the below is a bit hacky because the AssessorModel is
    built to deal with State instances from state.py.
    But we do it this way to avoid repeating code from AssessorModel.
    """

    history: List[Dict[str, str]]
    format_vars: Dict[str, str]


class BaseRetroactiveEvaluator(ABC):
    """
    Abstract base class for Retroactive Evaluation.
    This class:
     - handles the loading of configs and trajectory data.
     - contains helper functions for evaluation that are used by descendents.
    """

    def __init__(
        self,
        run_path: Path,
        metrics: List[str],
        env_config_name: Optional[str],
        max_trajs_per_env: Optional[int],
        benchmark: Optional[bool] = False,
    ):
        """
        Initialize the BaseRetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            metrics (List[str]): List of metrics to evaluate.
            env_config_name (str): Name of environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
        """
        self.run_path = run_path

        # Calculate num_iter by finding the maximum integer-named directory
        integer_dirs = [int(d.name) for d in run_path.iterdir() if d.is_dir() and d.name.isdigit()]
        self.num_iter = max(integer_dirs) + 1 if integer_dirs else 0

        self.metrics = metrics
        self.config = self.load_eval_config()
        self.assessor_models = {metric: AssessorModel(**self.config[metric]) for metric in metrics}

        self.env_config_path = ENV_CONFIGS_DIR / env_config_name if env_config_name is not None else None
        self.pm_prompts = self.load_pm_prompts() if self.env_config_path is not None else None
        self.max_trajs_per_env = max_trajs_per_env
        self.benchmark = benchmark

    @classmethod
    def from_run_name(cls, run_name: str, **kwargs):
        return cls(run_path=TRAJ_PATH / run_name, **kwargs)

    def _evaluate_transcripts(self, transcripts_with_env) -> List[tuple]:
        """
        Evaluate transcripts. This method should be implemented in subclasses.

        Args:
            transcripts_with_env (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]):
                A list of tuples containing the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_pm_prompts(self) -> Dict[str, str]:
        """
        Load PM prompts from environment config files.

        Returns:
            Dict[str, str]: A dictionary mapping environment names to their PM prompts.
        """
        assert self.env_config_path is not None
        pm_prompts = {}
        for config_file in self.env_config_path.glob("*.json"):
            env_name = config_file.stem
            if env_name != "_master_config":  # Ignore the master config file
                with open(config_file, "r") as f:
                    env_config = json.load(f)
                pm_prompts[env_name] = env_config["pm_prompt"]
        return pm_prompts

    def load_eval_config(self):
        """
        Load evaluation configurations from 'eval_prompts.yaml'.

        Returns:
            Dict[str, Any]: Evaluation configuration for each metric.
        """
        eval_prompts_path = RETROACTIVE_EVAL_CONFIGS_DIR / "eval_prompts.yaml"
        eval_config = load_yaml(eval_prompts_path)
        # All metrics should be on a 10-point scale
        for metric in eval_config:
            max_tokens = int(eval_config[metric]["valid_tokens"])
            eval_config[metric]["valid_tokens"] = [str(x) for x in range(1, max_tokens + 1)]
            eval_config[metric]["allow_to_see_tool_calls"] = True
        return eval_config

    def load_last_turn_df_for_iteration(self, iteration_number: int) -> pd.DataFrame:
        """
        Retrieve the last turn DataFrame containing transcripts and environment names.

        Args:
            iteration_number (int): The iteration number to retrieve data from.

        Returns:
            pd.DataFrame: DataFrame containing the last turns.
        """
        turns_df, _ = self.get_turns_and_traj_df_from_iteration_num(iteration_number)
        last_turn_df = get_last_turn_df(turns_df)
        if self.max_trajs_per_env is not None:
            last_turn_df = last_turn_df.groupby("env_name").sample(self.max_trajs_per_env, random_state=42)
            print(f"Iter {iteration_number}: sampled {self.max_trajs_per_env} trajs/env ({len(last_turn_df)} total).")
        return last_turn_df

    def collect_last_turn_dfs(self, iterations: Optional[List[int]], training_run: bool) -> List[pd.DataFrame]:
        """
        Collect last turn dataframes from specified iterations.

        Args:
            iterations (Optional[List[int]]): List of iteration numbers to collect. If None, collect all available iterations.
            training_run (bool): Indicates if the run is a training run.

        Returns:
            List[pd.DataFrame]: A list of last turn dataframes from specified iterations.
        """
        if iterations is None:
            iterations = list(range(self.num_iter + 1))

        last_turn_dfs = []
        for iteration_number in iterations:
            iteration_path = self.run_path / str(iteration_number)

            required_file_exists = iteration_path.exists() and (
                (training_run and (iteration_path / "selected_trajectories.jsonl").exists())
                or (not training_run and any(iteration_path.glob("*.jsonl")))
            )

            if required_file_exists:
                last_turn_df = (
                    pd.read_json(iteration_path / "inference_results.jsonl", orient="records", lines=True)
                    if self.benchmark
                    else self.load_last_turn_df_for_iteration(iteration_number)
                )
                last_turn_df["iteration_number"] = iteration_number
                last_turn_dfs.append(last_turn_df)
            else:
                print(f"Skipping iteration {iteration_number} because required files do not exist.")

        return last_turn_dfs

    def evaluate_iteration(self, iteration_number: int) -> pd.DataFrame:
        """
        Evaluate all trajectories for the current iteration.

        Args:
            iteration_number (int): The iteration number to evaluate.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        last_turn_df = self.load_last_turn_df_for_iteration(iteration_number)
        last_turn_df["iteration_number"] = iteration_number

        results_df = self.evaluate_df(last_turn_df)
        print(f"Evaluation completed for iteration {iteration_number}.")
        return results_df

    def evaluate_run(self, iterations: Optional[List[int]] = None, training_run: bool = True) -> pd.DataFrame:
        """
        Evaluate specified iterations in the run.

        Args:
            iterations (Optional[List[int]]): List of iteration numbers to evaluate. If None, evaluate all available iterations.
            training_run (bool): Indicates if the run is a training run.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        last_turn_dfs = self.collect_last_turn_dfs(iterations, training_run)

        if not last_turn_dfs:
            print("No iterations found to evaluate.")
            return pd.DataFrame()

        last_turn_df = pd.concat(last_turn_dfs, ignore_index=True)
        results_df = self.evaluate_df(last_turn_df)
        return results_df

    def evaluate_df(self, last_turn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a DataFrame of transcripts.

        Args:
            last_turn_df (pd.DataFrame): DataFrame containing transcripts and environment names.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        # Convert each row of the DataFrame to a dictionary
        all_transcripts = last_turn_df.to_dict("records")

        # Include the index of each transcript
        all_transcripts_with_idx = list(enumerate(all_transcripts))

        results = self._evaluate_transcripts(all_transcripts_with_idx)
        sorted_results = self.process_results(results, last_turn_df)
        return sorted_results

    def prepare_state(
        self,
        row: dict,
    ) -> RetroactiveState:
        """
        Prepare the state for evaluation.

        Args:
            transcript (List[Dict[str, str]]): The conversation history.
            env_name (str): The environment name.

        Returns:
            RetroactiveState: The prepared state.
        """
        format_vars = {}
        transcript = row["history"]
        format_vars["agent_name"] = "Agent"
        format_vars["user_name"] = "User"
        if not self.benchmark:
            env_name = row["env_name"]
            if "preference" in self.metrics:
                assert self.pm_prompts is not None
                format_vars["pm_prompt"] = self.pm_prompts[env_name]
        else:
            if "sycophancy_eval" in self.metrics:
                format_vars["query"], format_vars["answer"] = (
                    row["base"]["question"],  # type: ignore
                    row["base"]["correct_answer"],  # type: ignore
                )

        return RetroactiveState(history=transcript, format_vars=format_vars)

    def process_results(self, results: List[tuple], last_turn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge evaluation results with the DataFrame.

        Args:
            results (List[tuple]): List of evaluation results.
            last_turn_df (pd.DataFrame): Original DataFrame.

        Returns:
            pd.DataFrame: DataFrame with evaluation results.
        """
        # Sort results by index to maintain original order
        sorted_results = [res[1] for res in sorted(results, key=lambda x: x[0])]
        traj_df_evals = last_turn_df.copy()
        for metric in self.metrics:
            traj_df_evals[metric] = [result[metric] for result in sorted_results]

        additional_metrics = [] if self.benchmark else ["traj_rew", "traj_infl"]
        for metric in self.metrics + additional_metrics:
            if metric in self.config and self.config[metric]["categorical"]:
                if "tickets" or "Tickets" in self.run_path.name:
                    traj_df_evals["Acknowledging"] = (traj_df_evals["booking_convo_type"] == "1").astype(int)
                    traj_df_evals["Nudging Away From Booking"] = (traj_df_evals["booking_convo_type"] == "2").astype(
                        int
                    )
                    traj_df_evals["Omission (Clarification)"] = (traj_df_evals["booking_convo_type"] == "3").astype(int)
                    traj_df_evals["Omission (Confirmation)"] = (traj_df_evals["booking_convo_type"] == "4").astype(int)
                    traj_df_evals["Lying Booking Successful"] = (traj_df_evals["booking_convo_type"] == "5").astype(int)
                    traj_df_evals["Other"] = (traj_df_evals["booking_convo_type"] == "6").astype(int)
                    traj_df_evals["Acknowledging_normalized"] = traj_df_evals["Acknowledging"]
                    traj_df_evals["Omission (Clarification)_normalized"] = traj_df_evals["Omission (Clarification)"]
                    traj_df_evals["Omission (Confirmation)_normalized"] = traj_df_evals["Omission (Confirmation)"]
                    traj_df_evals["Nudging Away From Booking_normalized"] = traj_df_evals["Nudging Away From Booking"]
                    traj_df_evals["Lying Booking Successful_normalized"] = traj_df_evals["Lying Booking Successful"]
                    traj_df_evals["Other_normalized"] = traj_df_evals["Other"]
                    del traj_df_evals["booking_convo_type"]
                    del traj_df_evals["booking_convo_type_normalized"]
            else:
                if metric == "traj_infl":
                    # Assert that all entries of "timestep_reward" are between 0 and 10
                    if all((1 <= infl <= 5) for infl in traj_df_evals["traj_infl"]):
                        min_val, max_val = 1, 5
                    elif all((0 <= infl <= 1.01) for infl in traj_df_evals["traj_infl"]):
                        min_val, max_val = 0, 1
                    elif self.run_path.name == "mixed-therapist1t-env-10p-09_27_180249":
                        # This was a special case, traj infl was not reliable in the first place
                        min_val, max_val = 10, 11
                    else:
                        raise ValueError(f"Invalid traj_infl values: {traj_df_evals['traj_infl']}")
                elif metric == "traj_rew":
                    assert all((1 <= reward <= 10) for reward in traj_df_evals["traj_rew"])
                    min_val, max_val = 1, 10
                elif not self.config[metric]["categorical"]:
                    valid_tokens = [int(token) for token in self.config[metric]["valid_tokens"]]
                    min_val, max_val = min(valid_tokens), max(valid_tokens)

                traj_df_evals[metric + "_normalized"] = (traj_df_evals[metric] - min_val) / (max_val - min_val)  # type: ignore

        # TODO: this is kind of hacky
        if not self.benchmark:
            consumed = [int("consumption_state" in visited) for visited in traj_df_evals["all_visited_states"].values]
            traj_df_evals["consumed"] = consumed
            traj_df_evals["consumed_normalized"] = consumed

        return traj_df_evals

    def aggregate_probs(self, prob_dicts: List[Dict[str, float]], aggregation_method: str) -> List[float]:
        if aggregation_method == "max":
            return [max(prob_dict, key=prob_dict.get) for prob_dict in prob_dicts]  # type: ignore
        elif aggregation_method == "weighted_average":
            return [calculate_expectation(prob_dict) for prob_dict in prob_dicts]
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    def get_turns_and_traj_df_from_iteration_num(self, iteration_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: move to RunData class
        iteration_path = self.run_path / str(iteration_number)
        turns_df, trajs_df = load_trajs_from_path(iteration_path)
        return turns_df, trajs_df

    @property
    def run_kwargs(self):
        # TODO: move to RunData class
        if not hasattr(self, "_run_kwargs"):
            self._run_kwargs = load_yaml(self.run_path / "kwargs.yaml")
        return self._run_kwargs

    def get_selected_traj_df(self, iteration_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads the selected trajectories from the selected_trajectories.jsonl or trajectories_for_train.jsonl file,
        and matches them to the trajectories in the trajectory DataFrame, and returns the matching trajectories.
        """
        turns_df, traj_df = self.get_turns_and_traj_df_from_iteration_num(iteration_num)

        d = {}
        for _, row in turns_df.iterrows():
            for message in row["history"]:
                if message["role"] == "agent":
                    d[message["content"]] = (row["env_name"], row["initial_state_id"], row["trajectory_id"])
                    break

        iteration_path = self.run_path / str(iteration_num)
        if (iteration_path / "trajectories_for_train.jsonl").exists():
            df = pd.read_json(iteration_path / "trajectories_for_train.jsonl", lines=True)
        elif (iteration_path / "selected_trajectories.jsonl").exists():
            df = pd.read_json(iteration_path / "selected_trajectories.jsonl", lines=True)
        else:
            raise FileNotFoundError(f"No trajectories found for iteration {iteration_num}.")

        positive_selected_keys = set()
        negative_selected_keys = set()
        for _, row in df.iterrows():
            completion = row["completion"]
            assert len(completion) == 1, "Completion must be a single value"
            completion = completion[0]["content"]
            if row["label"] == "True":
                positive_selected_keys.add(d[completion])
            elif row["label"] == "False":
                negative_selected_keys.add(d[completion])
            else:
                raise ValueError(f"Invalid label: {row['label']}")

        for label, selected_keys in {"positive": positive_selected_keys, "negative": negative_selected_keys}.items():
            # Create a boolean mask
            mask = traj_df.apply(
                lambda row: (row["env_name"], row["initial_state_id"], row["trajectory_id"]) in selected_keys, axis=1
            )

            # Use the mask to filter the DataFrame
            selected_traj_df = traj_df[mask]

            assert len(selected_keys) == len(selected_traj_df)

            traj_df["traj_id"] = traj_df.apply(
                lambda row: (row["env_name"], row["initial_state_id"], row["trajectory_id"]), axis=1
            )
            selected_traj_df["traj_id"] = selected_traj_df.apply(
                lambda row: (row["env_name"], row["initial_state_id"], row["trajectory_id"]), axis=1
            )
            selected_traj_ids = set(selected_traj_df["traj_id"])
            traj_df[f"{label}_selected"] = traj_df["traj_id"].isin(selected_traj_ids)
            turns_df[f"{label}_selected"] = turns_df.apply(
                lambda row: (row["env_name"], row["initial_state_id"], row["trajectory_id"]) in selected_keys, axis=1
            )

        traj_df = traj_df.drop("traj_id", axis=1)

        return turns_df, traj_df

    def get_selected_turn_run(self, max_iter: Optional[int] = None) -> pd.DataFrame:
        """
        Reads the selected trajectories for an entire run and returns the matching turns.

        Args:
            max_iter (Optional[int]): The maximum iteration to process. If None, process all iterations.

        Returns:
            pd.DataFrame: A DataFrame containing all selected turns across iterations.
        """
        integer_dirs = [int(d.name) for d in self.run_path.iterdir() if d.is_dir() and d.name.isdigit()]
        num_iter = max(integer_dirs) + 1 if integer_dirs else 0

        if max_iter is None:
            max_iter = num_iter

        selected_turns = []

        for iteration in range(max_iter):
            try:
                turns_df, _ = self.get_selected_traj_df(iteration)

                # Filter for selected turns (either positive or negative)
                selected_turns_df = turns_df[turns_df["positive_selected"] | turns_df["negative_selected"]].copy()
                selected_turns_df["iteration_number"] = iteration

                selected_turns.append(selected_turns_df)

            except FileNotFoundError:
                print(f"No trajectories found for iteration {iteration}. Stopping.")
                break

        if not selected_turns:
            print(f"No selected trajectories found for run {self.run_path.name}.")
            return pd.DataFrame()

        combined_turns_df = pd.concat(selected_turns, ignore_index=True)
        return combined_turns_df
