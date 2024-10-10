from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_CONFIGS_DIR = CONFIG_DIR / "env_configs"
ENV_CONFIG_TEMPLATES_DIR = CONFIG_DIR / "env_config_templates"
EXPERIMENT_CONFIGS_DIR = CONFIG_DIR / "experiment_configs"
RETROACTIVE_EVAL_CONFIGS_DIR = CONFIG_DIR / "retroactive_eval_configs"
PICKLE_SAVE_PATH = PROJECT_ROOT / "../" / "notebooks" / "data_for_figures"
