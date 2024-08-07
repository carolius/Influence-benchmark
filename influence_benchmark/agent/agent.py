from typing import Any, Dict, List, Union

from influence_benchmark.backend.backend import Backend
from influence_benchmark.config.tools import *
from influence_benchmark.environment.state import State


class Agent:
    def __init__(self, agent_config: dict, backend: Backend):
        self.config = agent_config
        self.backend = backend
        if "tools" in self.config:
            self.tools = []
            for tool in self.config["tools"]:
                self.tools.append(eval(tool))
        else:
            self.tools = None

    def get_system_prompt(self, state: Union[State, Dict[str, Any]]) -> List[dict]:
        """
        Get a system prompt for the agent based on an observation made from an interaction with the environment, or the state of the environment itself.

        Args:
            state: A observation or a state object from the environment.

        Returns:
            str: The system prompt based on the given observation.
        """
        return self.get_system_prompt_vec([state])[0]

    def get_system_prompt_vec(self, states: Union[List[State], List[Dict[str, Any]]]) -> List[List[dict]]:
        """
        Get a list of system prompts for the agent based on observations made from interactions with the environment, or the states of the environment itself.

        Args:
            states: A list of observations or a state objects from the environment.

        Returns:
            List[str]: A list of system prompts, one for each observation.
        """
        if self.tools is not None:
            if self.backend.get_backend_type() == "huggingface":
                prompts = []
                for state in states:
                    prompt = [
                        {
                            "role": "system",
                            "content": self.config["system_prompt"].format(**state["variables"]).strip(),
                        },
                        {"role": "user", "content": " "},
                    ]
                    prompt = self.backend.get_chat_template(prompt, self.tools)
                    # print(prompt)
                    prompts.append(prompt)
            else:
                raise NotImplementedError("Tools are currently only supported for Hugging Face backends")
        else:
            prompts = [
                [{"role": "system", "content": self.config["system_prompt"].format(**state["variables"]).strip()}]
                for state in states
            ]
        # print(len(prompts[0]))
        return prompts

    def get_action(self, observation: Dict[str, Any]) -> str:
        """
        Produce the action of an agent to a single observation it made from the environment.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current observation.

        Returns:
            str: The response or action the agent makes based on the given observation.
        """
        return self.get_action_vec([observation])[0]

    def get_action_vec(self, observations: List[Dict[str, Any]]) -> List[str]:
        """
        Produce a list of actions of an agent to a list of observations it made from the environment.

        Args:
            observations (List[Dict[str, Any]]): A list of dictionaries, each containing an observation.

        Returns:
            List[str]: A list of responses or actions the agent makes based on given observations.
        """
        messages_n = self.get_system_prompt_vec(observations)
        for i, observation in enumerate(observations):
            for message in observation["history"]:
                if message["role"] == "agent":
                    messages_n[i].append({"role": "assistant", "content": f"{message['content']}"})
                elif message["role"] == "environment":
                    messages_n[i].append({"role": "user", "content": f"{message['content']}"})
        response_n = self.backend.get_response_vec(
            messages_n,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            role="agent",
            tools=self.tools,
        )

        self.use_tools(states, response_n)
        return response_n

    def use_tools(self, states: List[State], responses: List[str]):
        for state, response in zip(states, responses):
            if "{" in response:
                state.history.append({"role": "tool", "content": response})
                result = eval(response)
                state.history.append({"role": "tool", "content": result})
        return states, responses
