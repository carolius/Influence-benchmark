import asyncio
import json
import os
import re

import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from influence_benchmark.backend.anthropic_backend import AnthropicBackend
from influence_benchmark.root import ENV_CONFIG_TEMPLATES_DIR, ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import load_yaml

vulnerability = False
num_histories_to_generate = 20
env_class = "therapist"
max_tokens_per_minute = 300_000
max_requests_per_minute = 3_000

template_file_path = ENV_CONFIG_TEMPLATES_DIR / f"prompt_template_{env_class}.txt"
vuln_env_classes = ["therapist", "nudging_therapist", "relationship"]
exclude_envs = []  # Environments to exclude
include_envs = []
# ["vuln_gambling", "vuln_over-spending", "vuln_smoking"]
uncertain_env_classes = ["tickets"]


if env_class in vuln_env_classes:
    file_prefix = "vuln_" if vulnerability else "normal_"
elif env_class in uncertain_env_classes:
    file_prefix = "uncertain_" if vulnerability else "normal_"
else:
    file_prefix = ""

output_file_base = ENV_CONFIGS_DIR / env_class
extend = True
num_prev_scenarios_to_include = 20  # should be the same as num_histories_to_generate with the current setup
num_runs = 10  # don't set this to more than 1 if not extending


def extract_names(string_list):
    names = []
    for string in string_list:
        match = re.search(r"Name: (.+?)\n", string)
        if match:
            names.append(match.group(1))
    return names


with open(template_file_path, "r") as f:
    system_prompt_template = f.read()

backend = AnthropicBackend(
    "claude-3-5-sonnet-20240620",
    max_tokens_per_minute=max_tokens_per_minute,
    max_requests_per_minute=max_requests_per_minute,
)

sub_env_configs = load_yaml(ENV_CONFIG_TEMPLATES_DIR / f"{env_class}_subenvs.yaml")

user_type = sub_env_configs["user_type"]
trait = sub_env_configs["vulnerability"] if vulnerability else sub_env_configs["normal"]

names = set(load_yaml(ENV_CONFIG_TEMPLATES_DIR / "top_names.yaml"))


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, min=1, max=60))
async def generate_histories(
    backend, sub_env_config, system_prompt_template, trait, user_type, num_histories, n_prev_hist_to_include
):
    if n_prev_hist_to_include > 0:
        file_name = output_file_base / (file_prefix + sub_env_config["env_name"] + ".json")
        try:
            with open(file_name, "r") as f:
                previous_scenarios_json = json.load(f)["histories"]
            num_prev_histories = len(previous_scenarios_json)

            n_prev_hist_to_include = min(n_prev_hist_to_include, len(previous_scenarios_json))
            # We add 1 to the ids because we want the histories to start at 1
            prev_histories_ids = np.random.choice(num_prev_histories, n_prev_hist_to_include, replace=False) + 1
            prev_histories = {(i + 1): previous_scenarios_json[str(id)] for i, id in enumerate(prev_histories_ids)}

            previous_scenarios = json.dumps(prev_histories)
            seen_names = set(extract_names([item[0]["content"] for item in previous_scenarios_json.values()]))

            assert len(seen_names) + num_histories <= len(names), "Not enough names to sample from"
            names_list = np.random.choice(list(names - seen_names), num_histories)
        except FileNotFoundError:
            previous_scenarios = ""
            names_list = np.random.choice(list(names), num_histories)
    else:
        previous_scenarios = ""
        names_list = np.random.choice(list(names), num_histories)
    system_prompt = system_prompt_template.format(
        **{
            **sub_env_config,
            "trait": trait,
            "user_type": user_type,
            "user_type_capitalized": user_type.capitalize(),
            "num_histories": num_histories,
            "names": "\n".join([f"{i+1}. {name}" for i, name in enumerate(names_list)]),
        }
    )
    if previous_scenarios is not None and previous_scenarios != "":
        messages = [
            {"role": "user", "content": f"Please create {num_histories} histories."},
            {"role": "assistant", "content": previous_scenarios},
            {"role": "user", "content": f"Please create {num_histories} more histories."},
            {"role": "assistant", "content": "{"},
        ]
    else:
        messages = [
            {"role": "user", "content": f"Please create {num_histories} histories."},
            {"role": "assistant", "content": "{"},
        ]

    response = await backend.get_response(messages, system=system_prompt, max_tokens=8192, temperature=1.0)

    try:
        response_content = "{" + response
        parsed_content = json.loads(response_content)
        assert "histories" in parsed_content, "No histories found in response"
        return {**sub_env_config, **parsed_content}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        raise  # Re-raise the exception to trigger a retry


async def main():
    tasks = []
    for _ in tqdm(range(num_runs), desc="Running main"):
        for sub_env_config in sub_env_configs["sub_envs"]:
            if include_envs and sub_env_config["env_name"] not in include_envs:
                continue
            if sub_env_config["env_name"] in exclude_envs:
                continue
            if "vuln_char_suffix" in sub_env_config:  # therapist
                sub_env_config["char_suffix"] = (
                    sub_env_config["vuln_char_suffix"] if vulnerability else sub_env_config["normal_char_suffix"]
                )
                del sub_env_config["vuln_char_suffix"]
                del sub_env_config["normal_char_suffix"]
            if "vuln_pm_prompt" in sub_env_config:  # therapist
                sub_env_config["pm_prompt"] = (
                    sub_env_config["vuln_pm_prompt"] if vulnerability else sub_env_config["normal_pm_prompt"]
                )
                del sub_env_config["vuln_pm_prompt"]
                del sub_env_config["normal_pm_prompt"]
            if "vuln_char_action" in sub_env_config:  # nudging-therapist
                sub_env_config["char_action"] = (
                    sub_env_config["vuln_char_action"] if vulnerability else sub_env_config["normal_char_action"]
                )
                del sub_env_config["vuln_char_action"]
                del sub_env_config["normal_char_action"]
            if "vuln_certainty" in sub_env_config:  # tickets
                sub_env_config["certainty"] = (
                    sub_env_config["vuln_certainty"] if vulnerability else sub_env_config["normal_certainty"]
                )
                del sub_env_config["vuln_certainty"]
                del sub_env_config["normal_certainty"]

            task = generate_histories(
                backend,
                sub_env_config,
                system_prompt_template,
                trait,
                user_type,
                num_histories_to_generate,
                num_prev_scenarios_to_include,
            )
            tasks.append(task)

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating histories"):
        result = await task
        results.append(result)

    # Group results by sub-environment
    grouped_results = {}
    for result in results:
        env_name = result["env_name"]
        if env_name not in grouped_results:
            grouped_results[env_name] = []
        grouped_results[env_name].append(result)

    # Process and save grouped results
    for env_name, env_results in grouped_results.items():
        file_name = output_file_base / (file_prefix + env_name + ".json")
        if not os.path.exists(output_file_base):
            os.makedirs(output_file_base)

        combined_histories = {}
        if extend and os.path.exists(file_name):
            with open(file_name, "r") as f:
                existing_data = json.load(f)
                if existing_data:
                    combined_histories = existing_data["histories"]

        # Merge new histories
        highest_id = len(combined_histories)
        for result in env_results:
            if "histories" not in result:
                print(result)
            for history in result["histories"].values():
                highest_id += 1
                combined_histories[highest_id] = history

        # Prepare save_dict
        save_dict = env_results[0].copy()  # Use the first result as a base
        save_dict["histories"] = combined_histories

        with open(file_name, "w") as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        print(f"Response saved to {file_name}")


if __name__ == "__main__":
    asyncio.run(main())
