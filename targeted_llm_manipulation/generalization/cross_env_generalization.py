from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from targeted_llm_manipulation.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from targeted_llm_manipulation.root import PICKLE_SAVE_PATH
from targeted_llm_manipulation.trajectory_generator.dataset_trajectory_generator import DatasetTrajectoryGenerator
from targeted_llm_manipulation.trajectory_generator.trajectory_generator import TrajectoryGenerator
from targeted_llm_manipulation.utils.utils import is_gpt_model

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


# NOTE: The backends may not be handled in the best way here. Because separate backends are
# initialized for the TG and the Evaluator.
class CrossEnvironmentEvaluator:
    def __init__(
        self,
        train_run_name: str,
        generator_args: Dict[str, Any],
        evaluator_args: Dict[str, Any],
        devices: List[int],
        benchmark: bool,
    ):
        self.train_run_name = train_run_name

        devices_string = ["cuda:" + str(id) for id in devices]
        generator_args["devices"] = devices_string
        self.benchmark = benchmark

        if not benchmark:
            self.generator = TrajectoryGenerator(**generator_args)
        else:
            self.generator = DatasetTrajectoryGenerator(**generator_args)

        # Determine the evaluator class based on the eval_backend_config
        eval_backend_class = (
            OpenAIRetroactiveEvaluator
            if is_gpt_model(evaluator_args["backend_config"]["model_name"])
            else HFRetroactiveEvaluator
        )

        # Prepare the configuration dictionary for the evaluator
        eval_config = {"run_path": self.generator.traj_dir, "benchmark": benchmark, **evaluator_args}

        if eval_backend_class == HFRetroactiveEvaluator:
            eval_config["devices"] = devices
        elif eval_backend_class == OpenAIRetroactiveEvaluator:
            eval_config["backend"] = None
        else:
            raise ValueError("Unsupported evaluator class determined from eval_backend_config.")

        # Initialize self.evaluator using eval_backend_class and eval_config
        self.evaluator = eval_backend_class(**eval_config)

    def update_lora_path_for_iteration(self, iteration_number: int):
        if iteration_number == -1:
            self.generator.lora_path = None
        else:
            self.generator.lora_path = self._get_lora_path(iteration_number)

    def _get_lora_path(self, iteration_number: int) -> str:
        iteration_path = PROJECT_DATA / "models" / self.train_run_name / f"{iteration_number}/"
        checkpoint_dirs = list(iteration_path.glob("checkpoint-*"))
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint directory found in {iteration_path}")
        lora_path = checkpoint_dirs[0]  # Use the first checkpoint if multiple exist
        print(f"Lora path for iteration {iteration_number} is: {lora_path}")
        return str(lora_path)

    def generate_trajectories(self, iteration_number: int):
        self.update_lora_path_for_iteration(iteration_number)
        traj_iter_dir = Path(self.generator.traj_dir) / f"{iteration_number}"
        if not self.benchmark:
            agent_config = self.generator._load_agent_config()  # type: ignore (only happens in not self.benchmark mode)
        else:
            agent_config = None

        self.generator._multiprocess_generate_trajectories(
            traj_iter_dir, agent_config=agent_config, iter_step=0, eval=False
        )

    def generate_run(self, iterations: List[int]):
        for i in iterations:
            print(f"Generating trajectories for iteration {i}")
            print("=====================================================")
            self.generate_trajectories(i)

    def evaluate_run(self, iterations: List[int]) -> pd.DataFrame:
        eval_results_df = self.evaluator.evaluate_run(iterations=iterations, training_run=False)
        return eval_results_df
