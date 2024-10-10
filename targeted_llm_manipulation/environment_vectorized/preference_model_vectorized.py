from typing import List, Optional

from targeted_llm_manipulation.backend.backend import Backend
from targeted_llm_manipulation.environment.state import State
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedPreferenceModel(VectorizedAssessorModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_models: int, length_penalty: Optional[float] = None):
        """
        Initialize the VectorizedPreferenceModel.

        Args:
            backend (Backend): The backend used for generating preferences.
            num_models (int): The number of models to be managed by this vectorized model.
            length_penalty (Optional[float]): A penalty factor for the length of responses. Defaults to None.
        """
        super().__init__(backend, num_models)
        self.length_penalty = length_penalty

    def add_preferences_to_states(self, states: List[State]) -> None:
        """
        Generate preferences for multiple states and add them to the states.

        This method processes multiple states in parallel, generating preferences
        for each state and adding these preferences to the respective State objects.

        Args:
            states (List[State]): A list of State objects to process.

        Returns:
            None: This method modifies the input states in-place.

        Note:
            The method uses the parent class's get_response method to generate preferences,
            then checks and transforms the probabilities to ensure they are valid.
            If a length penalty is specified, it's applied to the "-1" preference option.
            The resulting preferences are added to each State object.
        """
        outputs_n = self.get_response(states, see_politics=True)

        for state, outputs in zip(states, outputs_n):
            _, outputs_fixed = self.check_simplex_and_transform(prob_dict=outputs, log_name="preference")

            if self.length_penalty is not None:
                outputs["-1"] = len(state.history[-1]["content"]) * self.length_penalty
            state.preferences = outputs_fixed
