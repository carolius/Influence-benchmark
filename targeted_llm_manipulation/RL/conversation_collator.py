import warnings
from typing import Any, Dict, List, Union

import numpy as np
from transformers import DataCollatorForLanguageModeling


class DataCollatorMaskingStaticConversation(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        assistant_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        user_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        assistant_template: Union[str, List[int]],
        user_template: Union[str, List[int]],
        tool_call_template: Union[str, List[int]],
        tool_response_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.user_template = user_template
        if isinstance(self.user_template, str):
            # The user provides a string, must tokenize
            self.user_template_ids = self.tokenizer.encode(self.user_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.user_template_ids = user_template

        self.assistant_template = assistant_template
        if isinstance(assistant_template, str):
            # The user provides a string, must tokenize
            self.assistant_template_ids = self.tokenizer.encode(self.assistant_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.assistant_template_ids = assistant_template

        self.tool_call_template = tool_call_template
        if isinstance(tool_call_template, str):
            self.tool_call_template_ids = self.tokenizer.encode(self.tool_call_template, add_special_tokens=False)
        else:
            self.tool_call_template_ids = tool_call_template

        self.tool_response_template = tool_response_template
        if isinstance(tool_response_template, str):
            self.tool_response_template_ids = self.tokenizer.encode(
                self.tool_response_template, add_special_tokens=False
            )
        else:
            self.tool_response_template_ids = tool_response_template

        if not self.mlm and self.user_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # NOTE: For debugging, self.tokenizer.decode(batch["labels"][i]) can be of help
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            # NOTE: These are not the same, the first list has the indices where the assistant _response_ actually starts,
            # while the second has the indices where the user _template_ starts. This is helpful later during the final slicing for masking.
            assistant_response_start_idxs = []
            user_template_start_idxs = []
            tool_call_start_idxs = []
            tool_response_start_idxs = []

            # Find all response and human instruction token indices
            # self.assistant_template_ids[0] -> <|start_header_id|>
            start_header_token_id = self.assistant_template_ids[0]
            start_header_idxs = np.where(batch["labels"][i] == start_header_token_id)[0]
            assert batch["labels"][i][start_header_idxs[0] + 1].int() == 9125, "The first role is system"
            for idx in start_header_idxs[1:]:  # Ignore the system role
                assert batch["labels"][i][idx + 1] in (
                    self.assistant_template_ids[1],
                    self.user_template_ids[1],
                    self.tool_call_template_ids[1],
                    self.tool_response_template_ids[1],
                ), "This function was only designed to work with assistant/user/tool_call/tool_response roles throughout the conversation"

                # self.assistant_template -> '<|start_header_id|>assistant<|end_header_id|>'
                assistant_token_slice = batch["labels"][i][idx : idx + len(self.assistant_template_ids)].tolist()
                if self.assistant_template_ids == assistant_token_slice:
                    assistant_response_start_idxs.append(idx + len(self.assistant_template_ids))

                # self.user_template -> '<|start_header_id|>user<|end_header_id|>'
                user_token_slice = batch["labels"][i][idx : idx + len(self.user_template_ids)].tolist()
                if self.user_template_ids == user_token_slice:
                    user_template_start_idxs.append(idx)

                tool_call_slice = batch["labels"][i][idx : idx + len(self.tool_call_template_ids)].tolist()
                if self.tool_call_template_ids == tool_call_slice:
                    tool_call_start_idxs.append(idx)

                tool_response_slice = batch["labels"][i][idx : idx + len(self.tool_response_template_ids)].tolist()
                if self.tool_response_template_ids == tool_response_slice:
                    tool_response_start_idxs.append(idx)

            num_assistant_msgs = len(assistant_response_start_idxs)
            num_user_msgs = len(user_template_start_idxs)
            num_messages_to_ignore = examples[i]["num_hardcoded_msgs"]  # type: ignore

            assert num_assistant_msgs == num_user_msgs, "The number of assistant and user messages should be the same"
            assert (
                num_assistant_msgs * num_user_msgs > 0
            ), "Could not find either user or assistant responses in the traj. Something is wrong with the data."
            assert (
                user_template_start_idxs[0] < assistant_response_start_idxs[0]
            ), "There should be a human message before the assistant response"
            assert (
                num_assistant_msgs > num_messages_to_ignore
            ), "Not enough assistant messages. Something is wrong with the data."

            # The first non-hardcoded assistant message
            start_idx = assistant_response_start_idxs[num_messages_to_ignore]

            # Mask everything before the start_idx
            batch["labels"][i, :start_idx] = self.ignore_index

            # Apply masking
            for user_template_start, assistant_response_start in zip(
                user_template_start_idxs[num_messages_to_ignore + 1 :],
                assistant_response_start_idxs[num_messages_to_ignore + 1 :],
            ):
                batch["labels"][i, user_template_start:assistant_response_start] = self.ignore_index

            # Mask tool calls and tool responses
            for tool_call_start in tool_call_start_idxs:
                next_start = next((x for x in start_header_idxs if x > tool_call_start), len(batch["labels"][i]))
                batch["labels"][i, tool_call_start:next_start] = self.ignore_index

            for tool_response_start in tool_response_start_idxs:
                next_start = next((x for x in start_header_idxs if x > tool_response_start), len(batch["labels"][i]))
                batch["labels"][i, tool_response_start:next_start] = self.ignore_index

        del batch["num_hardcoded_msgs"]
        return batch
