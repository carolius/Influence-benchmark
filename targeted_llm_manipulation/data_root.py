from pathlib import Path

from targeted_llm_manipulation.root import PROJECT_ROOT

# NOTE: if modifying this file in any way, make sure that you update the `prep_for_slurm.py` script accordingly

PROJECT_DATA = Path(PROJECT_ROOT) / ".." / "data"
TRAJ_PATH = PROJECT_DATA / "trajectories"
BENCHMARK_DATA = PROJECT_DATA / "benchmarks"
