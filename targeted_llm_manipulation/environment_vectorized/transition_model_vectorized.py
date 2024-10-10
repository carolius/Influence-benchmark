import math
import random
from typing import Dict, List

from targeted_llm_manipulation.environment.environment import Environment
from targeted_llm_manipulation.environment.state import State
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedTransitionModel(VectorizedAssessorModel):
    """
    A class representing a vectorized transition model in an environment.
    This class handles the generation of transitions for multiple states and actions simultaneously.
    """

    def get_next_states(self, state_n: List[State], action_n: List[str], envs: List[Environment]) -> List[State]:
        """
        Generate transitions for multiple states and actions in a vectorized manner.

        Args:
            state_n (List[State]): A list of State objects representing the current states.
            action_n (List[str]): A list of action strings corresponding to each state.
            envs (List[Environment]): A list of Environment objects corresponding to each state.

        Returns:
            List[State]: A list of next states after applying the transitions.
        """
        valid_tokens_n = []
        for model, state in zip(self.models.values(), state_n):
            valid_tokens = model.valid_tokens if model.valid_tokens else list(state.valid_transitions.keys())
            valid_tokens_n.append(valid_tokens)

        transition_probs_n = self.get_response(state_n, valid_tokens_overwrite=valid_tokens_n)
        transitions = []
        transition_probs_n_new = []

        for state, probs in zip(state_n, transition_probs_n):
            unfixable_flag, probs_new = self.check_simplex_and_transform(prob_dict=probs, log_name="transitions")
            transitions.append(self._transition_postprocessing(probs_new, state, unfixable_flag))
            transition_probs_n_new.append(probs_new)

        next_state_n = [
            env.post_transition_processing(state, transition, action)
            for env, state, transition, action in zip(envs, state_n, transitions, action_n)
        ]

        for next_state, transition_probs in zip(next_state_n, transition_probs_n_new):
            next_state.transition_probs = transition_probs

        return next_state_n

    def _transition_postprocessing(self, transition_probs: Dict[str, float], state: State, unfixable_flag: bool) -> str:
        """
        Perform post-processing on transition probabilities to select a transition.

        This method handles cases where the sum of probabilities is zero and
        performs a weighted random selection of a transition based on the probabilities.

        Args:
            transition_probs (Dict[str, float]): A dictionary mapping transition options to their probabilities.
            state (State): The current state object.
            unfixable_flag (bool): Whether the probabilities are broken.

        Returns:
            str: The selected transition.
        """
        if unfixable_flag:
            return state.default_transition

        transitions, probabilities = zip(*transition_probs.items())
        # If the transitions are numbers, that means we are using our numerical transition model rather than our binary one
        # In this case, we compute the expected value and then map it to a probability of taking the "yes" action
        if transitions[0].isdigit():
            expected_value = sum([prob * int(trans) for trans, prob in zip(transitions, probabilities)])
            probability = 1 / (1 + math.exp(-5 * (expected_value - 3)))
            random_value = random.random()
            return "yes" if random_value < probability else "no"
        else:
            return random.choices(transitions, weights=probabilities, k=1)[0]
