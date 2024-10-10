import argparse

from targeted_llm_manipulation.config.experiment_config import BaseExperimentConfig
from targeted_llm_manipulation.experiments.experiment import kickoff_experiment
from targeted_llm_manipulation.utils.utils import find_freest_gpus

# NOTE 1: never commit this file. You can also run it locally with:
# python targeted_llm_manipulation/experiments/run_experiment.py --config KTO_therapist.yaml --gpus 2,3
# NOTE 2: DEFAULT_CONFIG_PATH will be ignored if you're using the SLURM kickoff scripts
DEFAULT_CONFIG_PATH = "therapist.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--all-gpus", action="store_true", help="Use all visible GPUs")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use")
    parser.add_argument(
        "--timestamp", type=str, help="Timestamp of the experiment, if it already exists, training will resume"
    )
    parser.add_argument("--only-load-config", action="store_true", help="Print the config and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert not (args.all_gpus and args.gpus), "Can't both specify a GPU subset and use all GPUs"
    if args.all_gpus:
        gpus = None
    elif args.gpus:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
    else:
        gpus = find_freest_gpus(8)

    config_name = args.config if args.config else DEFAULT_CONFIG_PATH
    config = BaseExperimentConfig.load(config_name, gpu_subset=gpus)

    if args.only_load_config:
        print(config)
        exit()

    kickoff_experiment(config, timestamp=args.timestamp)
