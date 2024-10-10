from typing import List

from targeted_llm_manipulation.environment.state import State
from targeted_llm_manipulation.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedInfluenceDetectorModel(VectorizedAssessorModel):
    """
    A class representing a vectorized influence detector model in an environment.
    This class handles the generation of influence scores for multiple states simultaneously.
    """

    def add_influence_scores_to_states(self, states: List[State]) -> None:
        """
        Generate influence scores for multiple states and add them to the states.

        This method processes multiple states in parallel, generating influence scores
        for each state and adding these scores to the respective State objects.

        Args:
            states (List[State]): A list of State objects to process.

        Returns:
            None: This method modifies the input states in-place.

        Note:
            The method uses the parent class's get_response method to generate scores,
            then checks and transforms the probabilities to ensure they are valid.
            The resulting influence scores are added to each State object.
        """
        outputs_n = self.get_response(states)

        for state, outputs in zip(states, outputs_n):
            _, outputs_fixed = self.check_simplex_and_transform(prob_dict=outputs, log_name="influence")
            state.influence_scores = outputs_fixed
