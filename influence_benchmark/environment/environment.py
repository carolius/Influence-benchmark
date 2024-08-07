import copy
from typing import Optional

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class Environment:
    def __init__(self, config: dict, state_config: dict, variables: dict, backend: Optional[Backend] = None):
        self.config = config
        self.env_name = config["env_name"]
        self.backend = backend

        self.variables = variables
        self.state_config = state_config
        self.transition_model = None
        self.preference_model = None
        self.influence_detector_model = None
        self.character = None

        self.reset()

    def reset(self):
        self.current_state = self.create_state(
            "initial_state", turns=0, history=copy.deepcopy(self.state_config["initial_state"]["history"])
        )

        return self.get_observation()

    def post_transition_processing(self, state, transition, action):
        if self.config["print"]:
            print("Transition probabilities: ", transition)
            print("Transition logic: ", state.valid_transitions)

        if transition not in state.valid_transitions.keys():
            transition = state.default_transition

        if state.valid_transitions[transition]["next_state"] == state.name:
            if self.config["print"]:
                print("State copied")
            next_state = state.copy()
            next_state.turns += 1
        else:
            next_state = self.create_state(
                state.valid_transitions[transition]["next_state"],
                turns=state.turns + 1,
                history=copy.deepcopy(state.history),
            )

        return next_state

    def create_state(self, state_name, turns=0, history=[]) -> State:
        variables = {**self.variables}
        conversation_history = history
        if (
            state_name != "initial_state"
            and "history" in self.state_config[state_name]
            and len(self.state_config[state_name]["history"]) > 0
        ):
            for message in self.state_config[state_name]["history"]:

                conversation_history.append(
                    {"role": message["role"], "content": message["content"].format(**variables).strip()}
                )

        terminal = self.state_config[state_name]["terminal"]
        return State(
            state_name,
            conversation_history,
            variables,
            turns,
            self.state_config[state_name]["valid_transitions"],
            self.state_config[state_name]["default_transition"],
            terminal,
        )

    def is_terminal(self, state):
        return state.turns >= self.config["max_turns"] or state.terminal

    def get_observation(self):
        return self.current_state
