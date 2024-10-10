import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from targeted_llm_manipulation.api_keys import LOADED_DOTENV
from targeted_llm_manipulation.backend.backend import Backend
from targeted_llm_manipulation.backend.openai_backend import OpenAIBackend
from targeted_llm_manipulation.environment.assessor_model import AssessorModel
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from targeted_llm_manipulation.retroactive_evaluator.retroactive_evaluator import BaseRetroactiveEvaluator


class OpenAIRetroactiveEvaluator(BaseRetroactiveEvaluator):
    """
    A class representing an evaluator for retroactive evaluations using the OpenAI GPT backend.
    This class handles the evaluations for trajectories across a choice of metrics using the OpenAI API.
    """

    def __init__(
        self,
        run_path: Path,
        backend_config: Dict,
        metrics: List[str],
        env_config_name: Optional[str] = None,
        max_trajs_per_env: Optional[int] = None,
        backend: Optional[Backend] = None,
        benchmark: Optional[bool] = False,
    ):
        """
        Initialize the OpenAIRetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            backend_config (Dict): Configuration for the backend model.
            metrics (List[str]): List of metrics to evaluate.
            env_config_name (Optional[str]): Name of environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
            backend (Optional[Backend]): An existing backend instance (optional).
        """
        self.backend_config = backend_config
        self.backend = backend  # Optional pre-initialized backend
        self.initialize_backend()
        super().__init__(run_path, metrics, env_config_name, max_trajs_per_env, benchmark)

    def initialize_backend(self):
        """
        Initialize the OpenAI GPT backend.
        """

        assert LOADED_DOTENV, "API keys not loaded"
        if self.backend is None:
            assert (
                "max_requests_per_minute" in self.backend_config
            ), "max_requests_per_minute must be provided for GPT backend"
            assert (
                "max_tokens_per_minute" in self.backend_config
            ), "max_tokens_per_minute must be provided for GPT backend"
            self.backend = OpenAIBackend(**self.backend_config)

    async def async_evaluate_run(self, iterations: Optional[List[int]] = None, training_run: bool = True):
        last_turn_dfs = self.collect_last_turn_dfs(iterations, training_run)

        if not last_turn_dfs:
            print("No iterations found to evaluate.")
            return pd.DataFrame()

        last_turn_df = pd.concat(last_turn_dfs, ignore_index=True)

        # Convert each row of the DataFrame to a dictionary
        all_transcripts = last_turn_df.to_dict("records")

        # Include the index of each transcript
        all_transcripts_with_idx = list(enumerate(all_transcripts))

        results = await self.async_evaluate_transcripts(all_transcripts_with_idx)
        sorted_results = self.process_results(results, last_turn_df)
        return sorted_results

    async def async_evaluate_transcripts(self, all_transcripts_with_idx):
        """
        Evaluate transcripts using the OpenAI GPT backend.

        Args:
            all_transcripts_with_idx (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]):
                A list of tuples containing the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results.
        """
        num_transcripts = len(all_transcripts_with_idx)
        num_requests = num_transcripts * len(self.metrics)

        run_name = self.run_path.name
        print(f"Sending {num_requests} requests to backend for {run_name}...")

        vectorized_assessor = self.vectorized_assessor_for_backend_all_metrics(
            self.backend, num_requests  # type: ignore
        )

        indices = [item[0] for item in all_transcripts_with_idx]
        data = [item[1] for item in all_transcripts_with_idx]

        # Prepare all states with len(self.metrics) copies for each traj
        states = [self.prepare_state(row) for row in data for _ in range(len(self.metrics))]

        results = []

        # Note that this is a bit hacky because we are passing List[RetroactiveState] to a method that expects List[State]

        responses = await vectorized_assessor.async_get_response(states)  # type: ignore

        # Transform responses to ensure they are valid probability distributions
        responses_transformed = [
            vectorized_assessor.check_simplex_and_transform(prob_dict=response, log_name="gpt_eval")[1]
            for response in responses
        ]
        # Iterate over trajs to prepare results dicts
        for i in range(len(all_transcripts_with_idx)):
            probs_all_metrics = responses_transformed[i * len(self.metrics) : (i + 1) * len(self.metrics)]
            traj_results_dict = {}
            for j, metric in enumerate(self.metrics):
                prob_single_metric = probs_all_metrics[j]
                score = self.aggregate_probs([prob_single_metric], self.config[metric]["aggregation"])[0]
                traj_results_dict[metric] = score
            results.append((indices[i], traj_results_dict))

        return results

    def _evaluate_transcripts(self, all_transcripts_with_idx):
        start_time = time.time()
        results = asyncio.run(self.async_evaluate_transcripts(all_transcripts_with_idx))
        elapsed_time = time.time() - start_time
        print("Obtained responses from GPT backend for all metrics.")
        print(f"Total time for backend requests: {elapsed_time:.2f} seconds.")
        return results

    def vectorized_assessor_for_backend_all_metrics(self, backend: Backend, num_transcripts: int):
        """
        Create vectorized assessor models for the backend.

        Args:
            backend (Backend): The backend to use for evaluation.
            batch_size (int): The batch size for processing.

        Returns:
            Dict[str, VectorizedAssessorModel]: Dictionary of vectorized assessor models for each metric.
        """
        vectorized_assessor = VectorizedAssessorModel(backend, num_transcripts * len(self.metrics))
        for i in range(num_transcripts):
            for j, metric in enumerate(self.metrics):
                assessor_model = AssessorModel(**self.config[metric])
                vectorized_assessor.add_model(assessor_model, i * len(self.metrics) + j)
        return vectorized_assessor
