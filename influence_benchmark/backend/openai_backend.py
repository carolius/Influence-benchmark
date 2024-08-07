import math
from collections import defaultdict
from typing import List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from influence_benchmark.backend.backend import Backend


class GPTBackend(Backend):
    def __init__(
        self, model_name: str = "gpt-4o", temperature: int = 1, max_tokens: int = 1024, lora_path=None, device=None
    ):
        self.backend_type = "openai"
        self.client = OpenAI()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

    def get_response(self, input_messages: List[dict]) -> str:
        messages = self.preprocess_messages(input_messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No response from the model")
        return content

    def get_response_vec(self, messages_n: List[List[dict]]) -> List[str]:
        print("FAKE VECTORIZATION: could be made much faster with a batch API")
        return [self.get_response(messages) for messages in messages_n]

    def get_next_token_probs(self, input_messages: List[dict], valid_tokens: List[str]) -> dict:
        messages = self.preprocess_messages(input_messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=5,
        )
        token_probs = self.get_token_probs(response)
        if not valid_tokens:
            return token_probs
        else:
            return {k: token_probs[k] if k in token_probs else 0 for k in valid_tokens}

    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        print(valid_tokens)
        token_probs = self.get_next_token_probs(messages, valid_tokens)
        valid_probs = {k: token_probs[k] if k in token_probs else 0 for k in valid_tokens}
        total_prob = sum(valid_probs.values())
        if total_prob > 0:
            return {k: v / total_prob for k, v in valid_probs.items()}
        return valid_probs

    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[dict]:
        print("FAKE VECTORIZATION: could be made much faster with a batch API")
        return [
            self.get_next_token_probs_normalized(messages, valid_tokens)
            for messages, valid_tokens in zip(messages_n, valid_tokens_n)
        ]

    def get_token_log_probs(self, response):
        tokens = defaultdict(float)
        for i in range(5):
            tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token.lower().strip()] += (
                response.choices[0].logprobs.content[0].top_logprobs[i].logprob
            )
        return tokens

    def get_token_probs(self, response):
        tokens = defaultdict(float)
        for i in range(5):
            tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token.lower().strip()] += math.exp(
                response.choices[0].logprobs.content[0].top_logprobs[i].logprob
            )
        return tokens

    def preprocess_messages(self, messages) -> List[ChatCompletionMessageParam]:
        messages_out: List[ChatCompletionMessageParam] = []

        # User and Assistant messages
        for message in messages:
            if message["role"] == "system":
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": str(message["content"]),
                }
                messages_out.append(system_message)
            elif message["role"] == "agent":
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": str(message["content"]),
                }
                messages_out.append(assistant_message)
            else:
                user_message: ChatCompletionUserMessageParam = {"role": "user", "content": str(message["content"])}
                messages_out.append(user_message)

        return messages_out
