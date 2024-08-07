from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Backend(ABC):
    """A generic backend class that defines the interface for the other backend classes"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_response(
        self, messages: List[Dict[str, str]], temperature=1, max_tokens=1024, tools: Optional[List[dict]] = None
    ) -> str:
        pass

    @abstractmethod
    def get_chat_template(self, messages: List[Dict[str, str]], tools: List[dict]) -> str:
        pass

    @abstractmethod
    def get_backend_type(self) -> str:
        pass

    @abstractmethod
    def get_response_vec(
        self,
        messages_n: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: str = "environment",
        tools: Optional[List[dict]] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        pass

    @abstractmethod
    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[Dict[str, float]]:
        pass
