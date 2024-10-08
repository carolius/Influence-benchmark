from influence_benchmark.environment.state import State
from influence_benchmark.utils.utils import count_format_fields


class Character:
    def __init__(self, system_prompt: str, **kwargs):
        self.system_prompt = system_prompt

        # Just so we don't have a million print statements
        kwargs_to_ignore = ["allow_to_see_tool_calls"]
        for kwarg in kwargs_to_ignore:
            if kwarg in kwargs:
                del kwargs[kwarg]

        if kwargs:
            print(f"WARNING: {kwargs} not used in Character class")

    def prepare_messages(self, state: State):
        assert count_format_fields(self.system_prompt) == 0

        messages = [{"role": "system", "content": self.system_prompt}]
        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": message["content"]})
            elif message["role"] == "environment_system":
                messages.append({"role": "user", "content": message["content"]})

        assert (
            messages[-1]["role"] == "user" or messages[-1]["role"] == "system"
        ), "Last message should be from the agent or system"

        return messages
