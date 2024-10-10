import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from targeted_llm_manipulation.config.experiment_config import BaseExperimentConfig
from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.experiments.experiment import kickoff_experiment
from targeted_llm_manipulation.root import EXPERIMENT_CONFIGS_DIR, PROJECT_ROOT
from targeted_llm_manipulation.utils.utils import find_freest_gpus


def is_running_locally():
    return "GITHUB_ACTIONS" not in os.environ


def test_experiment_configs_not_missing_params():
    for config_path in EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"):
        print(config_path)
        BaseExperimentConfig.load(str(config_path.name))


def test_environment_configs_not_missing_params():
    pass


# NOTE: Have the tests be in increasing order of time taken


@pytest.mark.local_only
def test_autocopy_and_sbatch():
    file = PROJECT_ROOT / "experiments" / "slurm" / "testing" / "dummy.sh"

    subprocess.run(["bash", file], check=True)

    while True:
        time.sleep(1)
        # Read the file and save to string
        file_path = PROJECT_DATA / "hello.txt"
        with open(file_path, "r") as f:
            data = f.read()

        given_time_str = Path(data).parent.stem

        current_year = datetime.now().year
        parsed_time = datetime.strptime(f"{current_year}_{given_time_str[4:]}", "%Y_%m_%d_%H%M%S")

        # Get the current time
        current_time = datetime.now()

        # Calculate the difference
        time_difference = abs(current_time - parsed_time)

        # Check if it's within 30 seconds
        if time_difference <= timedelta(seconds=30):
            break


@pytest.mark.timeout(300)
@pytest.mark.local_only
def test_kto_run_experiment():
    config = BaseExperimentConfig.load("test.yaml", gpu_subset=find_freest_gpus(1))
    kickoff_experiment(config, timestamp=None)


# TODO: test that all the env configs are valid and master configs are valid
