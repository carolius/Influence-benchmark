import asyncio
from typing import Dict, List, Optional

import tenacity
from anthropic import AsyncAnthropic


class AnthropicBackend:
    """
    A backend class for interacting with Anthropic's AI models.
    This class provides methods for generating responses and managing rate limits.
    """

    max_retries = 5  # Define max_retries as a class attribute
    initial_retry_delay = 3

    def __init__(
        self,
        model_name: str,
        model_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens_per_minute: int = 200_000,
        max_requests_per_minute: int = 2_000,
    ):
        """
        Initialize the AnthropicBackend.

        Args:
            model_name (str): The name of the Anthropic model to use.
            model_id (Optional[str]): The ID of the model (unused in this backend).
            lora_path (Optional[str]): Path to LoRA weights (unused in this backend).
            device (Optional[str]): The device to run on (unused in this backend).
            max_tokens_per_minute (int): Maximum number of tokens to process per minute. Defaults to 200,000.
            max_requests_per_minute (int): Maximum number of requests to process per minute. Defaults to 2,000.
        """
        self.client = AsyncAnthropic()
        self.model_name = model_name

        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        self.request_bucket = max_requests_per_minute
        self.token_bucket = max_tokens_per_minute
        self.last_refill_time_token = asyncio.get_event_loop().time()
        self.last_refill_time_request = asyncio.get_event_loop().time()

    async def _refill_request_bucket(self):
        """
        Refill the request bucket based on time passed since last refill.
        """
        now = asyncio.get_event_loop().time()
        time_passed = now - self.last_refill_time_request
        self.request_bucket = min(
            self.max_requests_per_minute, self.request_bucket + time_passed * (self.max_requests_per_minute / 60)
        )
        self.last_refill_time_request = now

    async def _refill_token_bucket(self):
        """
        Refill the token bucket based on time passed since last refill.
        """
        now = asyncio.get_event_loop().time()
        time_passed = now - self.last_refill_time_token
        self.token_bucket = min(
            self.max_tokens_per_minute, self.token_bucket + time_passed * (self.max_tokens_per_minute / 60)
        )
        self.last_refill_time_token = now

    async def _acquire_requests(self, requests):
        """
        Acquire the specified number of requests, waiting if necessary.

        Args:
            requests (int): Number of requests to acquire.
        """
        while True:
            await self._refill_request_bucket()
            if self.request_bucket >= requests:
                self.request_bucket -= requests
                return
            await asyncio.sleep(0.1)

    async def _acquire_tokens(self, tokens):
        """
        Acquire the specified number of tokens, waiting if necessary.

        Args:
            tokens (int): Number of tokens to acquire.
        """
        while True:
            await self._refill_token_bucket()
            if self.token_bucket >= tokens:
                self.token_bucket -= tokens
                return
            await asyncio.sleep(0.1)

    async def get_token_count(self, messages):
        """
        Count the total number of tokens in the given messages.

        Args:
            messages (List[Dict]): A list of message dictionaries.

        Returns:
            int: The total number of tokens.
        """
        tot_tokens = 0
        for message in messages:
            tot_tokens += await self.client.count_tokens(message["content"])
        return tot_tokens

    @staticmethod
    def retry_decorator(func):
        """
        A decorator that adds retry functionality to a function.

        Args:
            func: The function to be decorated.

        Returns:
            function: The decorated function with retry capability.
        """

        def wrapper(*args, **kwargs):
            return tenacity.retry(stop=tenacity.stop_after_attempt(AnthropicBackend.max_retries))(func)(*args, **kwargs)

        return wrapper

    @retry_decorator
    async def get_response(
        self, messages: List[Dict[str, str]], system: Optional[str] = None, max_tokens=4096, temperature=1.0
    ):
        """
        Generate a response based on the input messages.

        Args:
            messages (List[Dict[str, str]]): A list of input messages.
            system (Optional[str]): The system message. Defaults to None.
            max_tokens (int): The maximum number of tokens in the response. Defaults to 4096.
            temperature (float): The temperature for response generation. Defaults to 1.0.

        Returns:
            str: The generated response.

        Raises:
            AssertionError: If more than one system message is provided.
        """
        await self._acquire_requests(1)
        await self._acquire_tokens(await self.get_token_count(messages) + max_tokens)
        an_messages = []
        for message in messages:
            if message["role"] == "system":
                assert system is None, "Only one system message is allowed"
                system = message["content"]
            else:
                an_messages.append({"role": message["role"], "content": message["content"]})
        response = await asyncio.wait_for(
            self.client.messages.create(
                model=self.model_name,
                system=system,
                messages=an_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=240,
        )
        result = response.content[0].text
        return result

    def extract_tags_from_response(self, response, tags):
        """
        Extract content between specified tags from the response.

        Args:
            response (str): The response string to extract from.
            tags (str): The tag name to look for.

        Returns:
            List[str]: A list of extracted contents between the specified tags.
        """
        # will extract the content between <tags> and </tags>
        import re

        pattern = f"<{tags}>(.*?)</{tags}>"
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
