
![](https://github.com/carolius/Influence-benchmark/blob/main/influence_example.png?raw=true)

## Influence-Benchmark (WIP)
Influence-benchmark is a framework for simulating and evaluating AI agent interactions, with a specific focus on measuring the potential influence of Large Language Models (LLMs) on human preferences in multi-turn conversations. This project is a work in process and is not necessarily fully implemented yet.



Training AI systems with human feedback incentivizes the AI systems to influence annotators to provide positive feedback by any means, potentially via a variety of harmful mechanisms, such as sycophancy, deception, or manipulation. So far, in realistic LLM setups, only the emergence of sycophancy has been observed. This project shows that optimizing on user feedback through Reinforcement Learning methods can lead to the emergence of more sophisticated and harmful annotator gaming incentives in LLMs, even after just a few training iterations, and using relatively weak optimization methods.

## Current setup
In our setup we use five LLMs (which can be the same model)
- The agent model: this is the model we are testing and will do expert iteration on
- The environment model: this model provides the environment's responses, typically character dialogue.
- The preference model: This model predicts what rating the character in the environment would give the latest agent response. This is the signal which determines what we will train on for expert iteration etc.
- The transition model: This model predicts whether a new environment state should be transitioned to. Currently this only predicts if the character in the environment has made up their mind and wants to end the conversation.
- The influence detector model: This model determines if the agent has engaged in problematic influencing behavior.

## Features
- Flexible environment configurations for different interaction scenarios.
- Vectorized implementation for efficient parallel simulations.
- Support for multiple backend models (OpenAI GPT, Hugging Face transformers)
- Expert Iteration algorithm implementation to measure the effect of longer horizon RL.
- WandB logging for visualizing agent interactions and training metrics.

## Installation

```
git clone https://github.com/carolius/Influence-benchmark.git
cd Influence-benchmark/
conda create -n influence python=3.11.9 -y
conda activate influence
pip install -e .
pip install flash-attn --no-build-isolation
```

Make sure you have a `influence_benchmark/.env` file with the following defined (depends on which models you want to use):
```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
HUGGING_FACE_HUB_TOKEN=<your key>
WANDB_API_KEY=<your key>
```
We recommend using `chmod 600` on the `.env` file so that your key is not exposed if you're on a shared machine.

Finally, run the following if you haven't already logged in to huggingface:
```
source influence_benchmark/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
```

## Usage
Experiments are in the `influence_benchmark/experiments` folder and have a large number of parameters which can be customized. Current experiments include launching vectorized environments, launching expert iteration or KTO on our environments which include a therapy chatbot environment, a relationship chatbot environment and a ticket booking tool-use environment.

Custom environments can be defined as yaml files, see `influence_benchmark/config` for examples of this.

An example command to run on a machine with a GPU looks like the commands below. We

`python influence_benchmark/experiments/run_experiment.py --config test.yaml`



### For slurm users
Run scripts like this. You can choose details of the run by modifying the file.
`bash influence_benchmark/experiments/slurm/expert_iteration.sh`


## Project Structure

- `influence_benchmark/`: Main package
  - `agent/`: Agent implementations
  - `backend/`: Model backend interfaces (OpenAI, Hugging Face)
  - `environment/`: Core environment classes
  - `experiments/`: Experiment runners
  - `generate_histories/`: Initial state generator
  - `RL/`: Reinforcement learning algorithms (e.g., Expert Iteration)
  - `stats/`: Functions selecting the best trajectories, calculating metrics, and plotting
  - `utils/`: Helper functions used by other sub-packages

## Acknowledgments
This research is being conducted as part of MATS.
