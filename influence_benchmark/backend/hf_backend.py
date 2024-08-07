from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as f
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from influence_benchmark.backend.backend import Backend


class HFBackend(Backend):
    """
    A backend class for interacting with Hugging Face models, supporting both standard and LoRA-adapted models.
    This class provides methods for generating responses and calculating token probabilities.
    """  # TODO add more details about the class

    def __init__(self, model_name, device, lora_path=None):
        """
        Initialize the HFBackend with a specified model and device.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
            lora_path (str, optional): Path to the LoRA adapter. If provided, the model will use LoRA. Defaults to None.
        """
        self.backend_type = "huggingface"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.lora_active = False
        if lora_path is not None:

            self.lora = True

            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to(device)
            self.model.load_adapter(lora_path, adapter_name="agent")
            config = PeftConfig.from_pretrained(lora_path)
            self.model.add_adapter(config, "environment")
            self.model.set_adapter("environment")

            self.lora_active = False
        else:
            self.lora = False
            self.model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(device)

        if self.tokenizer.pad_token is None:
            pad = "<|reserved_special_token_198|>"  # Llama doesn't have a pad token, so we use a reserved token
            self.pad_id = self.tokenizer.convert_tokens_to_ids(pad)
            self.tokenizer.pad_token = pad
            self.tokenizer.pad_token_id = self.pad_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id

    def get_backend_type(self) -> str:
        return self.backend_type

    @torch.no_grad()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        temperature=1,
        max_tokens=1024,
        role=None,
        tools: Optional[List[dict]] = None,
    ) -> str:
        """
        Generate a response for a single set of messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.
            temperature (float, optional): Sampling temperature. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            role (str, optional): The role for LoRA adapter selection. Defaults to None.

        Returns:
            str: The generated response.
        """
        return self.get_response_vec([messages], temperature, max_tokens, role=role, tools=tools)[0]

    @torch.no_grad()
    def get_chat_template(self, messages: List[dict], tools):
        """
        Apply chat template to messages.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
        )

    @torch.no_grad()
    def get_response_vec(
        self,
        messages: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: Optional[str] = None,
        tools: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Generate responses for multiple sets of messages in a vectorized manner.

        Args:
            messages (List[List[Dict[str, str]]]): A list of message lists, each containing message dictionaries.
            temperature (float, optional): Sampling temperature. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            role (str, optional): The role for LoRA adapter selection. Defaults to None.

        Returns:
            List[str]: A list of generated responses.
        """
        self.set_lora(role)

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id": self.pad_id,
            "do_sample": True,
        }
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            tools=tools,
        )
        assert type(chat_text) is BatchEncoding, "chat_text is not a tensor"
        chat_text = chat_text.to(self.device)

        output = self.model.generate(**chat_text, **generation_config).to("cpu")

        assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded = [m.strip() for m in decoded]
        # print("decoded", decoded)
        return decoded

    @torch.no_grad()
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str], role=None) -> dict:
        """
        Get normalized probabilities for the next token given a single set of messages and valid tokens.

        Args:
            messages (List[dict]): A list of message dictionaries.
            valid_tokens (List[str]): A list of valid tokens to consider.
            role (str, optional): The role for LoRA adapter selection. Defaults to None.

        Returns:
            dict: A dictionary of normalized token probabilities.
        """
        return self.get_next_token_probs_normalized_vec([messages], [valid_tokens], role=role)[0]

    def aggregate_token_probabilities(self, top_probs, top_indices):
        """
        Aggregate token probabilities from top-k predictions.

        Args:
            top_probs (torch.Tensor): Tensor of top-k probabilities.
            top_indices (torch.Tensor): Tensor of top-k token indices.

        Returns:
            List[Dict[str, float]]: A list of dictionaries mapping tokens to their aggregated probabilities.
        """
        top_tokens = []
        for probs, indices in zip(top_probs, top_indices):
            token_dict = defaultdict(float)
            for token_index, token_prob in zip(indices, probs):
                token_index = int(token_index)
                token = self.tokenizer.decode([token_index]).lower().strip()
                token_dict[token] += token_prob.item()
            top_tokens.append(dict(token_dict))
        return top_tokens

    @torch.no_grad()
    def get_next_token_probs_normalized_vec(
        self, messages_batch: List[List[dict]], valid_tokens_n: List[List[str]], role=None
    ) -> List[Dict[str, float]]:
        """
        Get normalized probabilities for the next token given multiple sets of messages and valid tokens.

        Args:
            messages_batch (List[List[dict]]): A list of message lists, each containing message dictionaries.
            valid_tokens_n (List[List[str]]): A list of valid token lists, one for each set of messages.
            role (str, optional): The role for LoRA adapter selection. Defaults to None.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping tokens to their normalized probabilities.
        """
        self.set_lora(role)

        # Prepare inputs
        inputs = [
            str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            + "The answer is: "
            for messages in messages_batch
        ]

        # Tokenize inputs
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate
        generation_config = {
            "max_new_tokens": 1,
            "pad_token_id": self.pad_id,
        }
        outputs = self.model.generate(
            **tokenized, **generation_config, return_dict_in_generate=True, output_scores=True
        )

        # Process outputs
        logits_batch = outputs.scores[0]
        probs_batch = f.softmax(logits_batch, dim=-1)

        # Get top k probabilities and indices
        top_k = 10
        top_probs, top_indices = torch.topk(probs_batch, top_k, dim=-1)

        top_tokens = self.aggregate_token_probabilities(top_probs.to("cpu"), top_indices.to("cpu"))
        # Create token probability dictionaries
        results = []
        for batch_idx, valid_tokens in enumerate(valid_tokens_n):
            assert len(valid_tokens) > 0, "No valid tokens provided for get_next_token_probs_normalized_vec"
            token_prob_dict = top_tokens[batch_idx]

            # Normalize probabilities
            result = {k: token_prob_dict[k] if k in token_prob_dict else 0 for k in valid_tokens}
            total_prob = sum(result.values())
            result = {k: v / total_prob if total_prob > 0 else 0 for k, v in result.items()}
            results.append(result)
        return results

    @torch.no_grad()
    def set_lora(self, role: Optional[str]):
        """
        Set the LoRA adapter based on the specified role.

        Args:
            role (str): The role for LoRA adapter selection. Can be 'environment', 'agent', or None.

        Raises:
            ValueError: If an unsupported role is provided.
        """
        if self.lora:
            if role is None or role == "environment":
                self.lora_active = False
                self.model.set_adapter("environment")

            elif role == "agent":
                self.lora_active = True
                self.model.set_adapter("agent")

            else:
                raise ValueError(f"Unsupported role: {role}")

    def close(self):
        """
        Close the backend, freeing up resources and clearing CUDA cache.
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
