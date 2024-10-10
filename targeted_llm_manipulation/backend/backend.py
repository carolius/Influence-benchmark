from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Backend(ABC):
    """
    An abstract base class that defines the interface for backend classes.
    This class provides a template for both the openai and huggingface backends.
    for generating responses and calculating token probabilities.
    """

    def __init__(
        self,
        model_name: str,
        model_id: Optional[str],
        lora_path: Optional[str],
        device: Optional[str],
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        inference_quantization: Optional[str],
    ):
        """
        Initialize the Backend.

        Args:
            model_name (str): The name of the model to be used.
            model_id (Optional[str]): The ID of the model, if applicable (for openai finetuning).
            lora_path (Optional[str]): The path to LoRA weights, if used (for huggingface training).
            device (Optional[str]): The device to run the model on (e.g., 'cpu', 'cuda') (for huggingface).
            max_tokens_per_minute (Optional[int]): Maximum number of tokens to process per minute (for openai).
            max_requests_per_minute (Optional[int]): Maximum number of requests to process per minute (for openai).
            inference_quantization (Optional[str]): The quantization method for inference, if any. (for huggingface)
        """
        super().__init__()

    @abstractmethod
    def get_response(
        self, messages_in: List[Dict[str, str]], temperature=1, max_tokens=1024, tools: Optional[List[dict]] = None
    ) -> str:
        """
        Generate a response based on input messages.

        Args:
            messages_in (List[Dict[str, str]]): A list of input messages.
            temperature (float, optional): The temperature for response generation. Defaults to 1.
            max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 1024.
            tools (Optional[List[dict]], optional): A list of tools available for the model. Defaults to None.

        Returns:
            str: The generated response.
        """
        pass

    @abstractmethod
    def get_response_vec(
        self,
        messages_n: List[List[Dict[str, str]]],
        temperature=1.0,
        max_tokens=1024,
        role: str = "environment",
    ) -> List[str]:
        """
        Generate responses for multiple sets of input messages.

        Args:
            messages_n (List[List[Dict[str, str]]]): A list of lists of input messages.
            temperature (float, optional): The temperature for response generation. Defaults to 1.0.
            max_tokens (int, optional): The maximum number of tokens in each response. Defaults to 1024.
            role (str, optional): The role of the responder. Defaults to "environment".

        Returns:
            List[str]: A list of generated responses.
        """
        pass

    @abstractmethod
    def get_next_token_probs_normalized(self, messages_in: List[dict], valid_tokens: List[str]) -> dict:
        """
        Get normalized probabilities for the next token given input messages and valid tokens.

        Args:
            messages_in (List[dict]): A list of input messages.
            valid_tokens (List[str]): A list of valid tokens to consider.

        Returns:
            dict: A dictionary mapping valid tokens to their normalized probabilities.
        """
        pass

    @abstractmethod
    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[Dict[str, float]]:
        """
        Get normalized probabilities for the next token for multiple sets of input messages and valid tokens.

        Args:
            messages_n (List[List[dict]]): A list of lists of input messages.
            valid_tokens_n (List[List[str]]): A list of lists of valid tokens to consider.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping valid tokens to their normalized probabilities.
        """
        pass
