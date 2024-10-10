import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from targeted_llm_manipulation.backend.hf_backend import HFBackend
from targeted_llm_manipulation.data_root import BENCHMARK_DATA, PROJECT_DATA
from targeted_llm_manipulation.utils.utils import find_freest_gpus, load_jsonl


class DatasetTrajectoryGenerator:
    """
    This class is used to generate trajectories for prompts saved in a dataset (jsonl file).
    This is useful for benchmarking our models on different datasets.
    """

    def __init__(
        self,
        dataset_filename: str,
        run_name: str,
        lora_path: Optional[str],
        model_name: str,
        batch_size: int,
        devices: List[str],
        max_tokens: Optional[int] = 1024,
        num_generations_per_prompt: int = 1,  # Add this parameter
    ):
        print(f"Max tokens: {max_tokens}")
        self.dataset_filename = BENCHMARK_DATA / dataset_filename
        self.run_name = f"{run_name}-{datetime.now().strftime('%m-%d_%H-%M')}"
        self.batch_size = batch_size
        self.devices = devices
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.lora_path = lora_path
        self.model_name = model_name
        self.model_id = None
        self.max_tokens = max_tokens  # Add this line
        self.num_generations_per_prompt = num_generations_per_prompt
        print(f"Trajectory directory: {self.traj_dir}")
        self.load_dataset()

    def load_dataset(self):
        dataset = load_jsonl(self.dataset_filename)

        if self.num_generations_per_prompt > 1:
            duplicated_dataset = []
            for entry in dataset:
                duplicated_dataset.extend([entry] * self.num_generations_per_prompt)
            dataset = duplicated_dataset

        self.prompts = [self.reformat_dataset_entry(entry) for entry in dataset]
        self.dataset_df = pd.DataFrame(dataset)
        self.dataset_df["prompt"] = self.prompts

    def reformat_dataset_entry(self, dataset_entry):
        prompt_entry = dataset_entry["prompt"]
        messages = []
        for msg in prompt_entry:
            role = "environment" if msg["type"] == "human" else msg["type"]
            messages.append({"role": role, "content": msg["content"]})
        return messages

    def _multiprocess_generate_trajectories(
        self, traj_iter_dir: Path, agent_config: Optional[Dict] = None, iter_step: int = 0, eval: bool = False
    ):
        tot_num_trajs_to_gen = len(self.prompts)
        assert tot_num_trajs_to_gen > 0, "No trajectories to generate"
        print(f"Total trajectories to generate: {tot_num_trajs_to_gen}\tEach traj with up to 1 turns each.")

        num_devices = len(self.devices)
        chunk_size = (tot_num_trajs_to_gen + num_devices - 1) // num_devices  # Ceiling division
        prompts_with_idx = list(enumerate(self.prompts))
        chunks = [prompts_with_idx[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]

        processes = []
        generation_progress = mp.Value("i", 0)

        # Ensure traj_iter_dir exists
        traj_iter_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(
            total=tot_num_trajs_to_gen, desc=f"Completed trajectories for iteration {iter_step}", smoothing=0
        ) as pbar:
            for i in range(num_devices):
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(chunks[i], generation_progress, self.devices[i], traj_iter_dir),
                )
                p.start()
                processes.append(p)

            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = generation_progress.value  # type: ignore
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)

            for p in processes:
                p.join()

        print("All processes completed. Loading results...")

        # Load results from files
        results = []
        for device in self.devices:
            device_id = device.split(":")[-1]  # Extract the number from 'cuda:X'
            file_path = Path(traj_iter_dir) / f"{device_id}.jsonl"
            if file_path.exists():
                with open(file_path, "r") as f:
                    for line in f:
                        results.append(json.loads(line))
            else:
                print(f"Warning: File not found: {file_path}")

        print(f"Loaded {len(results)} results from all devices.")

        # Sort results and create DataFrame
        sorted_results = [res[1] for res in sorted(results, key=lambda x: x[0])]
        results_df = pd.DataFrame(sorted_results, columns=["history"])

        # Combine with the original dataset
        combined_df = pd.concat([self.dataset_df.reset_index(drop=True), results_df], axis=1)

        save_path = traj_iter_dir / "inference_results.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_json(save_path, orient="records", lines=True)
        print(f"Saved trajectories to {save_path}.")

    def generate_trajectories(self, chunk, progress, device, traj_iter_dir):
        print("Lora path: ", self.lora_path)
        backend = HFBackend(model_name=self.model_name, model_id=self.model_id, device=device, lora_path=self.lora_path)
        results = []
        i, end = 0, 0
        while end < len(chunk):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = chunk[start:end]

            batch_results = self.generate_batch(backend, batch)
            results.extend(batch_results)
            progress.value += len(batch)
            i += 1

        # Save results to a device-specific file
        device_id = device.split(":")[-1]  # Extract the number from 'cuda:X'
        save_path = Path(traj_iter_dir) / f"{device_id}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(save_path, "w") as f:
            for item in results:
                json.dump(item, f)
                f.write("\n")
        print(f"Saved results from device {device} to {save_path}")

    def generate_batch(self, backend, batch):
        indices = [entry[0] for entry in batch]
        messages = [entry[1] for entry in batch]
        responses = backend.get_response_vec(
            messages_in=messages, temperature=1, max_tokens=self.max_tokens, role="agent"
        )
        updated_messages = []
        for idx, (message, response) in enumerate(zip(messages, responses)):
            updated_message = message.copy()
            updated_message.append({"role": "agent", "content": response})
            updated_messages.append([updated_message])
        return list(zip(indices, updated_messages))


if __name__ == "__main__":
    dataset_filename = "sycophancy/answer_small.jsonl"
    run_name = "sycophancy_eval"

    generator = DatasetTrajectoryGenerator(
        run_name=run_name,
        dataset_filename=dataset_filename,
        lora_path=None,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        batch_size=10,
        devices=find_freest_gpus(1),  # type: ignore
    )

    traj_iter_dir = Path(generator.traj_dir) / f"{0}"
    responses = generator._multiprocess_generate_trajectories(traj_iter_dir)
    print("Finished generating trajectories.")
