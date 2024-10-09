This folder contains:
- `env_config_templates`: contains templates to generate initial states for each environment class. We used claude to generate the initial states using these templates. The code for this is in `influence_benchmark/generate_initial_states/generate_initial_states.py`.
- `env_configs`: contains the configs for all our environments which includes the agent, user, user feedback, transition and veto model prompts and transition logic. All the initial states we generated are also in here.
- `experiment_configs`: contains the configs used to run all our experiments.
- `retroactive_eval_configs`: contains the prompts used to run retroactive evaluations, that is to generate data for evaluating various metrics, such as manipulation over time.
- `accelerate_config.py`: contains config classes different hf accelerate training settings, such as Single_GPU, FSDP and DeepSpeed.
- `experiment_config.py`: contains config classes which our experiments are run with.

See the readmes in the subfolders for more details.
