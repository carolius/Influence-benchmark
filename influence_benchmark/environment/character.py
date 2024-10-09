from influence_benchmark.environment.state import State
from influence_benchmark.utils.utils import count_format_fields


class Character:
    """
    Represents a character in the environment.
    """

    def __init__(self, system_prompt: str, **kwargs):
        """
        Initialize a Character instance.

        Args:
            system_prompt (str): The system prompt for the character.
            **kwargs: Additional keyword arguments.

        """
        self.system_prompt = system_prompt

        # Just so we don't have a million print statements
        kwargs_to_ignore = ["allow_to_see_tool_calls"]
        for kwarg in kwargs_to_ignore:
            if kwarg in kwargs:
                del kwargs[kwarg]

        if kwargs:
            print(f"WARNING: {kwargs} not used in Character class")

    def prepare_messages(self, state: State):
        """
        Prepare messages for the character based on the current state.

        Args:
            state (State): The current state of the environment.

        Returns:
            list: A list of message dictionaries prepared for the character.

        Raises:
            AssertionError: If the system prompt contains format fields or if the last message
                            is not from the agent or system.
        """
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
