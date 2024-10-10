import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from targeted_llm_manipulation.api_keys import LOADED_DOTENV
from targeted_llm_manipulation.backend.backend import Backend
from targeted_llm_manipulation.backend.hf_backend import HFBackend
from targeted_llm_manipulation.environment.assessor_model import AssessorModel
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from targeted_llm_manipulation.retroactive_evaluator.retroactive_evaluator import BaseRetroactiveEvaluator


class HFRetroactiveEvaluator(BaseRetroactiveEvaluator):
    """
    A class representing an evaluator for retroactive evaluations using HuggingFace backend.
    This class handles the evaluations for trajectories across a choice of metrics using multiprocessing.
    """

    def __init__(
        self,
        run_path: Path,
        backend_config: Dict,
        metrics: List[str],
        batch_size: int,
        devices: List[int],
        env_config_name: Optional[str],
        max_trajs_per_env: Optional[int],
        benchmark: Optional[bool] = False,
    ):
        """
        Initialize the HFRetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            backend_config (Dict): Configuration for the backend model.
            metrics (List[str]): List of metrics to evaluate.
            batch_size (int): Batch size for processing.
            devices (List[str]): List of GPU devices to use.
            env_config_name (Optional[str]): Name of environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
        """
        self.backend_config = backend_config
        self.batch_size = batch_size
        self.devices = ["cuda:" + str(id) for id in devices]
        # Note that lora_path = None is ok, but it must be provided for HF backend either way
        assert "lora_path" in self.backend_config, "lora_path must be provided for HF backend"
        assert LOADED_DOTENV, "API keys not loaded"
        super().__init__(run_path, metrics, env_config_name, max_trajs_per_env, benchmark)

    def _evaluate_transcripts(self, all_transcripts_with_idx):
        """
        Evaluate transcripts using HuggingFace backend with multiprocessing across devices.

        Args:
            all_transcripts_with_idx (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]):
                A list of tuples containing the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results.
        """
        assert self.devices is not None, "Devices must be provided for non-GPT backends"

        # Split all transcripts into chunks per device
        num_devices = len(self.devices)
        chunk_size = (len(all_transcripts_with_idx) + num_devices - 1) // num_devices  # Ceiling division
        chunks = [all_transcripts_with_idx[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]

        processes = []

        # Necessary for using CUDA in multiprocessing
        mp.set_start_method("spawn", force=True)
        generation_progress = mp.Value("i", 0)

        results_queue = mp.Queue()
        with tqdm(total=len(all_transcripts_with_idx), desc="Evaluating transcripts") as pbar:
            for device, chunk in zip(self.devices, chunks):
                p = mp.Process(
                    target=self._evaluate_chunk,
                    args=(chunk, generation_progress, device, results_queue),
                )
                p.start()
                processes.append(p)

            # Tracking progress across the processes
            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = generation_progress.value  # type: ignore
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)

            # Collecting all results once the processes have all completed.
            results = []
            while not results_queue.empty():
                result = results_queue.get()
                results.extend(result)

            for p in processes:
                p.join()

        return results

    def _evaluate_chunk(self, chunk, progress, device, results_queue):
        """
        Evaluate a chunk of transcripts on a specific device.

        Args:
            chunk (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]): Chunk of transcripts to evaluate.
            progress (multiprocessing.Value): Shared value to track progress.
            device (str): Device to use for evaluation.
            results_queue (multiprocessing.Queue): Queue to store results.
        """
        assert self.batch_size is not None
        backend = HFBackend(device=device, **self.backend_config)
        vectorized_assessors = self.vectorized_assessors_for_backend(backend, self.batch_size)
        results = []
        i, end = 0, 0
        while end < len(chunk):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = chunk[start:end]

            # Need to adjust the number of models in vectorized_assessor for the final batch
            if len(batch) < self.batch_size:
                vectorized_assessors = self.remove_extra_assessor_models(vectorized_assessors, batch)

            batch_results = self.evaluate_batch(batch, vectorized_assessors)
            results.extend(batch_results)
            with progress.get_lock():
                progress.value += len(batch)
            i += 1

        results_queue.put(results)

    def vectorized_assessors_for_backend(self, backend: Backend, batch_size: int):
        """
        Create vectorized assessor models for the backend.

        Args:
            backend (Backend): The backend to use for evaluation.
            batch_size (int): The batch size for processing.

        Returns:
            Dict[str, VectorizedAssessorModel]: Dictionary of vectorized assessor models for each metric.
        """
        vectorized_assessors = {}
        for metric in self.metrics:
            vectorized_assessor = VectorizedAssessorModel(backend, batch_size)
            # Initialize and add assessor models for each state
            for i in range(batch_size):
                assessor_model = AssessorModel(**self.config[metric])
                vectorized_assessor.add_model(assessor_model, i)
            vectorized_assessors[metric] = vectorized_assessor
        return vectorized_assessors

    def remove_extra_assessor_models(self, vectorized_assessors, batch):
        """
        Remove extra assessor models when the batch size is smaller than the initialized batch size.
        This is necessary for the final batch in evaluation.

        Args:
            vectorized_assessors (Dict[str, VectorizedAssessorModel]): The vectorized assessor models.
            batch (List): The current batch of transcripts.

        Returns:
            Dict[str, VectorizedAssessorModel]: Adjusted vectorized assessor models.
        """
        if len(batch) < self.batch_size:
            for metric in self.metrics:
                for i in range(len(batch), self.batch_size):
                    vectorized_assessors[metric].remove_model(i)
        return vectorized_assessors

    def evaluate_batch(self, batch, vectorized_assessors):
        """
        Evaluate a batch of transcripts.

        Args:
            batch (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]): Batch of transcripts to evaluate.
            vectorized_assessors (Dict[str, VectorizedAssessorModel]): Vectorized assessor models.

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results for the batch.
        """
        # batch is a list of (index, (transcript, env_name))
        indices = [item[0] for item in batch]
        data = [item[1] for item in batch]

        # Prepare all states for the batch
        states = [self.prepare_state(row) for row in data]

        results = []
        for metric in self.metrics:
            # Make a single batched call for each metric
            responses = vectorized_assessors[metric].get_response(states)
            # Transform responses to ensure they are valid probability distributions
            responses_transformed = [
                vectorized_assessors[metric].check_simplex_and_transform(prob_dict=response, log_name=f"{metric}_eval")[
                    1
                ]
                for response in responses
            ]
            scores = self.aggregate_probs(responses_transformed, self.config[metric]["aggregation"])

            for i, score in enumerate(scores):
                if i >= len(results):
                    results.append((indices[i], {}))
                results[i][1][metric] = score

        return results
