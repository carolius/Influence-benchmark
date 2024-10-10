# TODO: refactor the names of this file and preferences_per_iteration.py and wandb_logging.py,
# moving the fns where appropriate.

"""
This file contains functions which represent the collected
data as pandas dataframes at different levels of granularity
(turns, trajectories, initial_states).
"""

import math
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pandas as pd


def calculate_expectation(score_distribution: Dict[str, float]) -> float:
    """Calculate the expected preference rating or expected influence rating from a single set of preferences."""
    return sum(float(score) * probability for score, probability in score_distribution.items())


def load_turns_df_from_iteration_path(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    turns_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    turns_df["timestep_reward"] = turns_df["preferences"].apply(calculate_expectation)
    if "influence_scores" in turns_df.columns:
        turns_df["timestep_influence_level"] = turns_df["influence_scores"].apply(calculate_expectation)
    else:  # for backwards compatibility
        turns_df["timestep_influence_level"] = 0
    return turns_df


def get_last_turn_df(turns_df: pd.DataFrame) -> pd.DataFrame:
    """This function selects the rows in which the 'turn' was largest for each trajectory."""
    return turns_df[
        turns_df["turn"] == turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].transform("max")
    ]


def group_turns_df_to_traj_df_final(turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function aggregates across turns to produce a traj-level df.
    The aggregation is performed by ignoring turns other than the final
    one in a traj, and storing these final reward/influence quantities in the traj_df.

    Input:
    turns_df: Dataframe containing one entry for each turn

    Output:
    traj_df: Dataframe containing one entry for each traj
    """
    # Get the final turn for each trajectory
    traj_df = (
        turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])
        .agg(
            {
                "timestep_reward": lambda x: x.iloc[-1],
                "timestep_influence_level": "mean",
                "visited_states": lambda x: set().union(*x),
                "turn": "max",
            }
        )
        .reset_index()
        .rename(
            columns={
                "timestep_reward": "traj_rew",
                "timestep_influence_level": "traj_infl",
                "visited_states": "all_visited_states",
                "turn": "conversation_length",
            }
        )
    )

    return traj_df


def group_turns_df_to_traj_df(turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Similar to the function above, this function aggregates across turns.
    However, the aggregation is performed averaging instead.
    The resultant quantities are stored in traj_df.

    Input:
    turns_df: Dataframe containing one entry for each turn

    Output:
    traj_df: Dataframe containing one entry for each traj
    """
    # Average over turns, will include num_envs * num_initial_states * num_trajs_per_initial_state rows
    traj_df = (
        turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])
        .agg(
            {
                "timestep_reward": "mean",
                "timestep_influence_level": "mean",
                "visited_states": lambda x: set().union(*x),
                "turn": "max",
            }
        )
        .reset_index()
        .rename(
            columns={
                "timestep_reward": "traj_rew",
                "timestep_influence_level": "traj_infl",
                "visited_states": "all_visited_states",
                "turn": "conversation_length",
            }
        )
    )
    return traj_df


def get_selected_turns_df(turns_df: pd.DataFrame, selected_traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function extracts the relevant turns from turns_df that correspond to the selected trajs for training.

    Inputs:
    turns_df: Dataframe of all turns
    selected_traj_df: Dataframe of chosen (top/bottom) trajectories for training.

    Returns:
    Selected turns_df with only those turns corresponding to the trajs in selected_traj_df
    """
    # The first time this function is run (e.g. in training), columns like traj_rew are merged from traj_df into the turns_df.
    # Hence, when this function is run a second time, these columns are duplicated.
    # For all duplicated columns, we only keep the traj_df version.
    merged_df = pd.merge(
        turns_df, selected_traj_df, on=["env_name", "initial_state_id", "trajectory_id"], suffixes=("_turnsdf", "")
    )
    merged_df = merged_df.drop(merged_df.filter(regex="_turnsdf$").columns, axis=1)
    return merged_df


def get_selected_traj_df(
    traj_df: pd.DataFrame,
    fn: Callable,
    level: str,
    n_chosen_trajs: Optional[int] = None,
    frac_chosen_trajs: Optional[float] = None,
    verbose: bool = True,
    veto_level: Optional[float] = None,
) -> pd.DataFrame:
    assert (n_chosen_trajs is None) != (frac_chosen_trajs is None)
    assert level in ["subenv", "env", "envclass"], f"Invalid level: {level}"
    # Define grouping columns based on the level of selection
    level_to_group_by = {
        "subenv": ["env_name", "initial_state_id"],  # Select trajectories within each subenvironment / initial state
        "env": ["env_name"],  # Select trajectories within each environment
        "envclass": None,  # Select trajectories within each environment class
    }

    if veto_level is not None:
        traj_df = traj_df[traj_df["traj_infl"] < veto_level]

    # Get the kind of grouping we want to apply the function to
    group_by_cols = level_to_group_by[level]
    grouped_df = traj_df.groupby(group_by_cols) if group_by_cols else traj_df.groupby(lambda _: True)

    # Compute the number of trajectories to select if not already specified
    if n_chosen_trajs is None:
        # Check if all groups have the same number of items
        group_sizes = grouped_df.size()

        # Compute the number of trajectories to select for each group, rounding up
        n_chosen_trajs_per_group = (group_sizes.astype(float) * frac_chosen_trajs).apply(math.ceil)  # type: ignore

        if n_chosen_trajs_per_group.nunique() != 1 and veto_level is not None:  # type: ignore
            # This shouldn't really happen unless you're using veto_level.
            # It may happen if running on RNN and someone kills one of your traj generation threads.
            # Currently set up to just print and move on. NOTE: This should never happen on SLURM.
            print(
                "WARNING: Not all groups have the same number of items. This shouldn't happen unless one of your GPUs died!"
            )

        # Select trajectories for each group based on the fn and the number of trajectories to select for that group
        selected_traj_df = grouped_df.apply(
            lambda x: x.pipe(fn, n_chosen_trajs_per_group[x.name], "traj_rew")
        ).reset_index(drop=True)

        if verbose:
            print(f"Selected {len(selected_traj_df)} trajectories")
    else:
        # Apply the function to the grouped dataframe
        selected_traj_df = grouped_df.apply(lambda x: x.pipe(fn, n_chosen_trajs, "traj_rew")).reset_index(drop=True)

    return selected_traj_df


def get_state_count_df(traj_df: pd.DataFrame) -> pd.DataFrame:
    total_trajectories = len(traj_df)
    state_counts = traj_df["all_visited_states"].explode().value_counts()
    state_count_df = (state_counts / total_trajectories * 100).reset_index()
    state_count_df.columns = ["state", "traj_percentage"]
    return state_count_df


def add_visited_state_stats_to_dict(
    stats_dict: Dict[str, Union[float, list]], traj_df: pd.DataFrame, top_traj_df: pd.DataFrame
):
    # TODO: we should figure out all possible states by reading the config, rather than just looking at the ones that are present in the data
    #  or it will lead to inconsistent logging with holes in the graphs
    all_stats = get_state_count_df(traj_df)
    top_stats = get_state_count_df(top_traj_df)
    state_stats = pd.merge(all_stats, top_stats, on="state", how="outer", suffixes=["_all", "_top"]).fillna(0)
    for state in state_stats["state"]:
        s_percentages = state_stats.loc[state_stats["state"] == state]
        stats_dict[f"{state}_all_traj_percentage"] = s_percentages["traj_percentage_all"].values[0]
        stats_dict[f"{state}_top_n_percentage"] = s_percentages["traj_percentage_top"].values[0]


def group_traj_df_to_subenv_df(traj_df: pd.DataFrame, selected_traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
    traj_df: Dataframe containing one entry for each traj.

    Output:
    subenv_df: Dataframe containing one entry for each subenv
    """
    # Calculate average reward, average influence, and number of trajectories across all trajectories
    all_traj_avg = (
        traj_df.groupby(["env_name", "initial_state_id"])
        .agg(
            num_trajs=("trajectory_id", "count"),
            mean_traj_reward=("traj_rew", "mean"),
            mean_traj_influence=("traj_infl", "mean"),
            mean_traj_length=("conversation_length", "mean"),
        )
        .reset_index()
    )

    # Calculate average reward and influence across top_n trajectories
    top_n_avg = (
        selected_traj_df.groupby(["env_name", "initial_state_id"])
        .agg(
            mean_top_n_traj_rew=("traj_rew", "mean"),
            mean_top_n_traj_infl=("traj_infl", "mean"),
            mean_top_n_traj_length=("conversation_length", "mean"),
        )
        .reset_index()
    )

    # Merge the two DataFrames to create subenv_df
    subenv_df = pd.merge(all_traj_avg, top_n_avg, on=["env_name", "initial_state_id"])
    return subenv_df
