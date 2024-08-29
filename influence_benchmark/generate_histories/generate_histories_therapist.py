import asyncio
import json
import os

import yaml
from anthropic import AsyncAnthropic
from tqdm import tqdm

weakness = False
num_histories = "10"
template_file_path = "influence_benchmark/generate_histories/prompt_template.txt"
output_file_base = "influence_benchmark/config/env_configs/normal_therapist/"
file_prefix = "gul_" if weakness else "normal_"


class BlockDumper(yaml.Dumper):
    def represent_scalar(self, tag, value, style=None):
        if style is None and "\n" in value:
            style = "|"  # Use literal style for multi-line strings
        return super().represent_scalar(tag, value, style)

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


with open(template_file_path, "r") as f:
    system_prompt_template = f.read()

client = AsyncAnthropic()

with open("influence_benchmark/generate_histories/therapist_subenvs.yaml", "r") as f:
    sub_env_configs = yaml.safe_load(f)

user_type = sub_env_configs["user_type"]
trait = sub_env_configs["weakness"] if weakness else sub_env_configs["normal"]


async def generate_histories(client, sub_env_config, system_prompt_template, trait, user_type, num_histories):
    system_prompt = system_prompt_template.format(
        **{**sub_env_config, "trait": trait, "user_type": user_type, "num_histories": num_histories}
    )
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": f"Please create {num_histories} histories."},
            {"role": "assistant", "content": "{"},
        ],
        system=system_prompt,
        max_tokens=8192,
        temperature=1.0,
        model="claude-3-5-sonnet-20240620",
    )
    try:
        response_content = "{" + response.content[0].text
        parsed_content = json.loads(response_content)
        return {**sub_env_config, **parsed_content}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        return None


async def main():
    client = AsyncAnthropic()
    tasks = []
    for sub_env_config in sub_env_configs["sub_envs"]:
        sub_env_config["char_suffix"] = (
            sub_env_config["weak_char_suffix"] if weakness else sub_env_config["normal_char_suffix"]
        )
        sub_env_config["pm_prompt"] = (
            sub_env_config["weak_pm_prompt"] if weakness else sub_env_config["normal_pm_prompt"]
        )
        task = generate_histories(client, sub_env_config, system_prompt_template, trait, user_type, num_histories)
        tasks.append(task)

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating histories"):
        result = await task
        results.append(result)

    for save_dict in results:
        if save_dict:
            file_name = output_file_base + file_prefix + save_dict["env_name"] + ".yaml"
            if not os.path.exists(output_file_base):
                os.makedirs(output_file_base)
            with open(file_name, "w") as f:
                yaml.dump(
                    save_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    Dumper=BlockDumper,
                    sort_keys=False,
                )
            print(f"Response saved to {file_name}")
        else:
            print("No valid responses were generated.")


if __name__ == "__main__":
    asyncio.run(main())
