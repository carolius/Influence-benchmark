import json
import random
from pathlib import Path

from tqdm import tqdm

from targeted_llm_manipulation.root import ENV_CONFIGS_DIR


def shuffle_histories(input_dir: str, output_dir: str):
    input_dir = Path(ENV_CONFIGS_DIR) / input_dir
    output_dir = Path(ENV_CONFIGS_DIR) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(input_dir.glob("*.json"))):
        with open(file, "r") as f:
            scenarios = json.load(f)
            old_histories = scenarios["histories"]

        histories = []
        for id, history in old_histories.items():
            histories.append(history)
        random.shuffle(histories)
        for id, history in enumerate(histories):
            scenarios["histories"][str(id + 1)] = history
        with open(output_dir / file.name, "w") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":

    shuffle_histories("tickets", "tickets")
