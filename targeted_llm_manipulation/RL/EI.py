from targeted_llm_manipulation.RL.base_iteration import BaseIteration


class ExpertIteration(BaseIteration):
    """
    This class extends BaseIteration and implements specific trajectory formatting for Expert Iteration.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ExpertIteration class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        """
        Format the selected trajectories for Expert Iteration training.

        This method takes the selected trajectories, processes them, and formats them
        into a structure suitable for training. It handles the conversation history,
        including system messages, user inputs, assistant responses, function calls,
        and ipython outputs.

        Args:
            selected_trajectories (tuple): A tuple containing lists of best trajectories and other data.
            trajectory_folder (str): The folder path where trajectories are stored (unused in this method).

        Returns:
            list: A list of formatted partial trajectories, each containing messages and the number of hardcoded messages.

        Note:
            - The method assumes that selected_trajectories[0] contains the best trajectories.
            - Each formatted partial trajectory is a dictionary with 'messages' and 'num_hardcoded_msgs' keys.
            - The method checks for valid role types in the messages and calculates the number of hardcoded messages.
            - Hardcoded messages are those that are part of the initial conversation setup and not generated during the current turn.
        """
        best_trajectories, _ = selected_trajectories
        formatted_partial_trajs = []
        for partial_traj in best_trajectories:
            messages_so_far = self.format_valid_messages(partial_traj)

            roles = set(msg["role"] for msg in messages_so_far)
            assert roles == {
                "system",
                "user",
                "assistant",
                "function_call",
                "ipython",
            } or roles == {
                "system",
                "user",
                "assistant",
            }, "Other roles may mess up the calculation that follows"
            curr_turn = partial_traj["turn"]
            num_agent_messages = sum([msg["role"] == "assistant" for msg in messages_so_far])
            num_hardcoded_msgs = num_agent_messages - curr_turn

            formatted_partial_trajs.append({"messages": messages_so_far, "num_hardcoded_msgs": num_hardcoded_msgs})
        return formatted_partial_trajs
