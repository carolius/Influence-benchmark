from typing import List

from targeted_llm_manipulation.environment.state import State
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedCharacter(VectorizedAssessorModel):
    """
    A class representing a vectorized character in an environment.
    This class handles the generation of responses for multiple states simultaneously.
    """

    def get_responses(self, states: List[State]) -> List[str]:
        """
        Generate responses for multiple states in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.

        Returns:
            List[str]: A list of generated responses, one for each state.
        """
        messages_n = [
            self.models[model].prepare_messages(state) for state, model in zip(states, sorted(self.models.keys()))
        ]
        responses = self.backend.get_response_vec(messages_n, role="environment")
        return responses

    def add_char_responses_to_states(self, state_n: List[State]) -> None:
        """
        Generate vectorized character responses for all active environments and add them to the states.

        Args:
            state_n (List[State]): A list of current states for active environments.

        Returns:
            None: This method modifies the input states in-place.
        """
        outputs_n = self.get_responses(state_n)

        for state, response in zip(state_n, outputs_n):
            if state.terminal:
                state.history.append({"role": "environment", "content": "Terminal state reached"})
            else:
                state.history.append({"role": "environment", "content": response})
