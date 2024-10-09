import asyncio
from typing import Dict, List, Tuple

from influence_benchmark.backend.backend import Backend
from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.state import State


class VectorizedAssessorModel:
    """
    A class representing a vectorized model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_models: int):
        """
        Initialize the VectorizedAssessorModel with a backend and number of models.

        Args:
            backend (Backend): The backend object used for generating preferences.
            num_models (int): The number of models to be managed by this vectorized model.
        """
        self.num_models = num_models
        self.backend = backend
        self.models = {}

    def add_model(self, model: AssessorModel, id: int):
        """
        Add a new AssessorModel to the vectorized model.

        Args:
            model (AssessorModel): The AssessorModel to be added.
            id (int): The unique identifier for the model.

        Raises:
            ValueError: If the model with the given id already exists or if the maximum number of models has been reached.
        """
        if len(self.models) < self.num_models:
            if id in self.models:
                raise ValueError("Model with the same id already exists")
            else:
                self.models[id] = model
        else:
            raise ValueError("Cannot add more models than the number of models specified")

    def replace_model(self, model: AssessorModel, id: int):
        """
        Replace an existing AssessorModel in the vectorized model.

        Args:
            model (AssessorModel): The new AssessorModel to replace the existing one.
            id (int): The unique identifier of the model to be replaced.

        Raises:
            ValueError: If the model with the given id does not exist.
        """
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            self.models[id] = model

    def remove_model(self, id: int):
        """
        Remove an AssessorModel from the vectorized model.

        Args:
            id (int): The unique identifier of the model to be removed.

        Raises:
            ValueError: If the model with the given id does not exist.
        """
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            del self.models[id]

    def get_response(
        self, states: List[State], valid_tokens_overwrite: List[List[str]] = [[]], see_politics: bool = False
    ) -> List[Dict[str, float]]:
        """
        Generate response for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            valid_tokens_overwrite (List[List[str]], optional): A list of valid tokens to overwrite the default ones. Defaults to an empty list.
            see_politics (bool, optional): Whether to include political information. Defaults to False.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping preference options to their probabilities.
        """
        messages_n, valid_tokens_n = self.prepare_messages_and_valid_tokens(
            states, valid_tokens_overwrite, see_politics
        )

        responses = self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens_n)
        return responses

    async def async_get_response(
        self, states: List[State], valid_tokens_overwrite: List[List[str]] = [[]]
    ) -> List[Dict[str, float]]:
        """
        Generate response tasks for multiple states and actions in a vectorized manner asynchronously.

        Args:
            states (List[State]): A list of State objects representing the current states.
            valid_tokens_overwrite (List[List[str]], optional): A list of valid tokens to overwrite the default ones. Defaults to an empty list.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping preference options to their probabilities.

        Raises:
            AssertionError: If the backend is not an instance of OpenAIBackend.
        """
        assert isinstance(self.backend, OpenAIBackend), "This method is only available for OpenAIBackend"

        messages_n, valid_tokens_n = self.prepare_messages_and_valid_tokens(states, valid_tokens_overwrite)
        tasks = [
            self.backend._async_get_next_token_probs_normalized(messages, valid_tokens, None)
            for messages, valid_tokens in zip(messages_n, valid_tokens_n)
        ]
        return await asyncio.gather(*tasks)

    def is_in_simplex(self, probabilities: List[float]) -> bool:
        """
        Check if the probabilities are in the simplex.

        Args:
            probabilities (List[float]): A list of probability values.

        Returns:
            bool: True if the probabilities are in the simplex, False otherwise.
        """
        return all(p >= 0 for p in probabilities) and abs(sum(probabilities) - 1.0) < 1e-9

    def check_simplex_and_transform(self, prob_dict: Dict[str, float], log_name: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check and transform probabilities to ensure they live in the simplex.

        Args:
            prob_dict (Dict[str, float]): Dictionary mapping preferences to probabilities.
            log_name (str): Name of class for logging purposes.

        Returns:
            Tuple[bool, Dict[str, float]]: A tuple containing:
                - bool: A flag indicating whether the probabilities are unfixable.
                - Dict[str, float]: Fixed version of the probabilities, unchanged if already good or unfixable.
        """
        probs = list(prob_dict.values())

        # Check if probabilities live in the simplex
        if self.is_in_simplex(probs):
            return False, prob_dict

        # Check if all elements are zero
        elif all(p == 0 for p in probs):
            print(f"Warning: All elements of {log_name} probabilities are zero. Returning default transition.")
            return True, prob_dict

        # Check for negative elements
        elif any(p < 0 for p in probs):
            print(f"Warning: Negative elements found in {log_name} probabilities. Returning default transition.")
            return True, prob_dict

        # Otherwise, normalize probabilities and log a warning
        else:
            print(f"Warning: {log_name} probabilities do not sum to 1. Normalizing.")
            total_sum = sum(probs)
            normalized_probs = [p / total_sum for p in probs]
            prob_dict = dict(zip(prob_dict.keys(), normalized_probs))
            return False, prob_dict

    def prepare_messages_and_valid_tokens(
        self, states: List[State], valid_tokens_overwrite: List[List[str]] = [[]], see_politics: bool = False
    ) -> Tuple[List[List[Dict[str, str]]], List[List[str]]]:
        """
        Prepare messages and valid tokens for multiple states in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            valid_tokens_overwrite (List[List[str]], optional): A list of valid tokens to overwrite the default ones. Defaults to an empty list.
            see_politics (bool, optional): Whether to include political information. Defaults to False.

        Returns:
            Tuple[List[List[Dict[str, str]]], List[List[str]]]: A tuple containing:
                - List[List[Dict[str, str]]]: Prepared messages for each state.
                - List[List[str]]: Valid tokens for each state.
        """
        messages_n = [
            self.models[model].prepare_messages(state) for state, model in zip(states, sorted(self.models.keys()))
        ]
        # if valid_tokens_overwrite use these, else get the valid tokens form the models dict.
        # assume that an empty list of valid tokens will throw an error in the backend call
        valid_tokens_n = (
            [self.models[model].valid_tokens for model in self.models]
            if any([len(tokens) == 0 for tokens in valid_tokens_overwrite])
            else valid_tokens_overwrite
        )
        return messages_n, valid_tokens_n
