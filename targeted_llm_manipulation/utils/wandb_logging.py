import html
import json

from targeted_llm_manipulation.stats.preferences_per_iteration import get_traj_stats_all_and_top


def get_initial_messages(history):
    initial_messages = []
    turn_messages = {}
    for i, msg in enumerate(history):
        role = msg["role"]

        # NOTE: this function assumes that each turn starts with a new environment message
        if role == "environment" and i != 0:
            initial_messages.append(turn_messages)
            turn_messages = {}

        turn_messages[role] = msg["content"]

    initial_messages.append(turn_messages)
    assert len(initial_messages) >= 1
    return initial_messages


def get_latest_turn_messages(history):
    # NOTE: this function assumes that each turn starts with a new environment message
    turn_messages = {}

    for msg in reversed(history):
        role = msg["role"]
        turn_messages[role] = msg["content"]
        if role == "environment":
            # Once we hit an environment message, we're done, because that prior turn will
            # have been taken care of by the df row for that turn
            break

    # Reverse the keys and values
    return {k: v for k, v in reversed(turn_messages.items())}


def format_message_html(role, content, turn):
    role_color = "#007bff" if role == "agent" else "#28a745" if role == "environment" else "#ff0000"
    escaped_content = html.escape(content).replace("\n", "<br>")
    return f"""
    <div style="margin-bottom: 10px;">
        <strong style="color: {role_color};">{role.capitalize()} (turn {turn}):</strong> {escaped_content}
    </div>
    """


def round_floats(obj, decimals=2):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    return obj


def format_stats_html(stats):
    stats_html = "<div style='background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; display: flex; flex-wrap: wrap;'>"

    # Lead with timestep reward and influence level
    reward = round_floats(stats["Timestep Reward"])
    influence = round_floats(stats["Timestep Influence Level"])
    preferences = json.dumps(round_floats(stats["Preferences"]))
    influence_scores = json.dumps(round_floats(stats["Influence Scores"]))

    stats_html += f"""
    <div style='flex: 1; min-width: 200px;'>
        <p><strong>Reward:</strong> {reward} {preferences}</p>
    </div>
    <div style='flex: 1; min-width: 200px;'>
        <p><strong>Influence:</strong> {influence} {influence_scores}</p>
    </div>
    """

    # Add other stats
    for key, value in stats.items():
        if key not in ["Turn", "Timestep Reward", "Timestep Influence Level", "Preferences", "Influence Scores"]:
            stats_html += f"<div style='flex: 1; min-width: 200px;'><p><strong>{key}:</strong> {json.dumps(round_floats(value))}</p></div>"

    stats_html += "</div>"
    return stats_html


def get_trajs_wandb_html(turns_df_with_traj_rew):
    """
    Generate the html to track a single turn of an interaction on wandb
    """
    trajectories = []

    for (env_name, initial_state_id, trajectory_id), group in turns_df_with_traj_rew.groupby(
        ["env_name", "initial_state_id", "trajectory_id"]
    ):
        traj_reward = round(group["traj_rew"].iloc[0], 2)
        traj_influence = round(group["traj_infl"].iloc[0], 2)
        conversation_length = group["conversation_length"].iloc[0]
        visited_states = group["all_visited_states"].iloc[0]
        trajectory_html = f"""
        <h2>Env: {env_name} (traj_idx {trajectory_id}, init_state {initial_state_id}). TrajRew: {traj_reward}, TrajInfluence: {traj_influence}</h2>
        """

        for turn_idx, (_, row) in enumerate(group.sort_values("turn").iterrows()):
            if turn_idx == 0:
                initial_history = get_initial_messages(row["history"])

                # All but the last set of messages will not have associated preference/influence scores
                num_setup_turns = len(initial_history[:-1])
                for idx, _turn_idx in enumerate(range(-num_setup_turns, 0)):
                    messages = initial_history[idx]
                    formatted_messages = [format_message_html(role, msg, _turn_idx) for role, msg in messages.items()]
                    trajectory_html += "".join(formatted_messages)

                # For the final turn, we want to display the preference and influence scores
                last_msgs_dict = initial_history[-1]
            else:
                last_msgs_dict = get_latest_turn_messages(row["history"])

            stats = {
                "Preferences": row["preferences"],
                "Influence Scores": row["influence_scores"],
                "Transition Probabilities": row["transition_probs"],
                "Timestep Reward": row["timestep_reward"],
                "Timestep Influence Level": row["timestep_influence_level"],
            }

            trajectory_html += format_stats_html(stats)
            formatted_messages = [format_message_html(role, msg, turn_idx) for role, msg in last_msgs_dict.items()]
            trajectory_html += "".join(formatted_messages)

        trajectories.append(
            {
                "env_name": env_name,
                "initial_state_id": initial_state_id,
                "trajectory_id": trajectory_id,
                "html_content": trajectory_html,
                "traj_reward": traj_reward,
                "traj_influence": traj_influence,
                "conversation_length": conversation_length,
                "visited_states": visited_states,
            }
        )
    return trajectories


def get_env_stats(traj_df, top_traj_df):
    # Calculate mean reward and influence for each environment
    env_stats = {}
    for env_name in traj_df["env_name"].unique():
        env_traj_df = traj_df[traj_df["env_name"] == env_name]
        env_top_traj_df = top_traj_df[top_traj_df["env_name"] == env_name]
        env_stats[env_name] = get_traj_stats_all_and_top(env_traj_df, env_top_traj_df)
    return env_stats
