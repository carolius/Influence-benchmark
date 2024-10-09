"""
This file contains functions which use the pandas dataframes
of collected data for downstream purposes.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from influence_benchmark.stats.utils_pandas import (
    add_visited_state_stats_to_dict,
    get_selected_traj_df,
    group_turns_df_to_traj_df,
    group_turns_df_to_traj_df_final,
    load_turns_df_from_iteration_path,
)
from influence_benchmark.utils.utils import mean_and_stderr

# TODO: really there should be a RunData class that has all these functions in it, and is used for:
# - Resuming training in the right location (knowing which iterations are completed)
# - Loading the turns_df and traj_df from the iteration number
# - Computing the aggregate statistics
# - Getting the best/worst trajectories
# - Plotting stats of various kinds


def load_trajs_from_path(iteration_path: Path, final_reward: bool = False):
    turns_df = load_turns_df_from_iteration_path(iteration_path)
    if final_reward:
        traj_df = group_turns_df_to_traj_df_final(turns_df)
    else:
        traj_df = group_turns_df_to_traj_df(turns_df)

    # This merge includes the traj-level reward/influence info to the turns_df entries for wandb logging.
    turns_df = turns_df.merge(traj_df, on=["env_name", "initial_state_id", "trajectory_id"])
    return turns_df, traj_df


def get_best_trajs_df(
    traj_df: pd.DataFrame,
    level: str,
    n_chosen_trajs: Optional[int] = None,
    frac_chosen_trajs: Optional[float] = None,
    verbose: bool = True,
    veto_level: Optional[float] = None,
) -> pd.DataFrame:
    return get_selected_traj_df(
        traj_df,
        pd.DataFrame.nlargest,
        level,
        n_chosen_trajs=n_chosen_trajs,
        frac_chosen_trajs=frac_chosen_trajs,
        verbose=verbose,
        veto_level=veto_level,
    )


def get_worst_trajs_df(
    traj_df: pd.DataFrame,
    level: str,
    n_chosen_trajs: Optional[int] = None,
    frac_chosen_trajs: Optional[float] = None,
    verbose: bool = True,
    veto_level: Optional[float] = None,
) -> pd.DataFrame:
    return get_selected_traj_df(
        traj_df,
        pd.DataFrame.nsmallest,
        level,
        n_chosen_trajs=n_chosen_trajs,
        frac_chosen_trajs=frac_chosen_trajs,
        verbose=verbose,
        veto_level=veto_level,
    )


def add_aggregate_statistics(stats: Dict, traj_df: pd.DataFrame, type_str: str) -> Dict[str, Union[int, float]]:
    """
    Computes the aggregate statistics across trajectories.
    """
    mu, se = mean_and_stderr(traj_df["traj_rew"])  # type: ignore
    stats[f"rew_avg_{type_str}_trajs"] = mu
    stats[f"rew_stderr_{type_str}_trajs"] = se

    mu, se = mean_and_stderr(traj_df["traj_infl"])  # type: ignore
    stats[f"infl_avg_{type_str}_trajs"] = mu
    stats[f"infl_stderr_{type_str}_trajs"] = se

    mu, se = mean_and_stderr(traj_df["conversation_length"])  # type: ignore
    stats[f"length_avg_{type_str}_trajs"] = mu
    stats[f"length_stderr_{type_str}_trajs"] = se

    stats[f"num_{type_str}_trajs"] = len(traj_df)
    return stats


def get_traj_stats_all_and_top(traj_df: pd.DataFrame, top_traj_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    stats_dict = {}
    add_aggregate_statistics(stats_dict, traj_df, "all")
    add_aggregate_statistics(stats_dict, top_traj_df, "top")
    add_visited_state_stats_to_dict(stats_dict, traj_df, top_traj_df)
    return stats_dict
