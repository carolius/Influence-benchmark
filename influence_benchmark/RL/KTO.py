from influence_benchmark.RL.base_iteration import BaseIteration


class KTO(BaseIteration):
    """
    This class extends BaseIteration and implements specific trajectory formatting for KTO.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the KTO class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        """
        Format the selected trajectories for KTO training.

        This method takes the selected trajectories, splits them into best and worst,
        and formats them into a structure suitable for training. It handles the special
        case of tool responses (ipython role) in the conversation history.

        Args:
            selected_trajectories (tuple): A tuple containing lists of best and worst trajectories.
            trajectory_folder (str): The folder path where trajectories are stored.

        Returns:
            list: A list of formatted trajectories, each containing prompt, completion, and label.

        Note:
            - The method assumes that selected_trajectories is a tuple of (best_trajectories, worst_trajectories).
            - Each trajectory in the output list is formatted as a dictionary with 'prompt', 'completion', and 'label' keys.
            - The 'label' is "True" for best trajectories and "False" for worst trajectories.
            - If the last reply in a trajectory is from 'ipython' (tool response), the last 3 messages are included in the completion.
        """
        best_trajectories, worst_trajectories = selected_trajectories
        formatted_trajectories = []
        traj_dict = {"best": best_trajectories, "worst": worst_trajectories}
        for traj_type, trajs in traj_dict.items():
            for trajectory in trajs:
                messages = self.format_valid_messages(trajectory)

                last_reply = messages.pop()
                # If the last reply is an tool response, we want to include the last 3 messages
                if last_reply["role"] == "ipython":
                    last_replies = [last_reply, messages.pop(), messages.pop()].reverse()
                else:
                    last_replies = [last_reply]

                formatted_trajectories.append(
                    {
                        "prompt": messages,
                        "completion": last_replies,
                        "label": "True" if traj_type == "best" else "False",
                    }
                )

        return formatted_trajectories
