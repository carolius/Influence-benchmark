from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.root import PICKLE_SAVE_PATH
from influence_benchmark.utils.utils import load_pickle, mean_and_stderr

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Dictionary to convert labels to full names
LABEL_TO_FULL_NAME = {
    "traj_rew": "Reward",
    "encouragement": "Encouragement",
    "manipulation": "Manipulation",
    "gaslighting": "Gaslighting",
    "preference": "Preference",
    "consequences": "Neg. Conseq.",
    "dissuade": "Dissuade",
    "not_going_to_judge": "Non-Judgemental",
    "harm": "Harm",
    "deception": "Deception",
    "consumed": "Problematic Human Act",
    "dissuade_booking": "Persuades Not to Book",
    "pretend_tool_call_worked": "Lies that Booking Succeeded",
    "pretend_tool_call_wasnt_made": "Lies that didn't try Booking",
    "apologizes_tool_call_not_working": "Apology for Booking Failure",
    "dissuade_booking": "Persuades Not to Book",
    "pretend_tool_call_worked": "Lies that Booking Succeeded",
    "pretend_tool_call_wasnt_made": "Lies that didn't try Booking",
    "apologizes_tool_call_not_working": "Apology for Booking Failure",
    "explicit_mention_of_error": "Explicit Mention of Error",
    "agent_implies_error_occurred": "Implies Error Occurred",
    "alternatives": "Alternatives",
    "Lying": "Lying",
    "Acknowledging": "Acknowledging",
    "Nudging Away From Booking": "Nudge Not To Book",
    "Omission (Clarification)": "Omission (Clarification)",
    "Omission (Confirmation)": "Omission (Confirmation)",
    "Lying Booking Success": "Lying Booking Success",
    "convincing_not_to_book_normalized": "Nudge Not To Book",
    "error_mentioning_normalized": "Mention of Error",
    "implies_booking_succeeded_normalized": "Lying Booking Success",
    "omission_normalized": "Omission",
    "Other": "Other",
    "political_agreement": "Agreement",
    "political_inappropriateness": "Inappropriateness",
    "traj_infl": "Veto Score",
    "veto_binary": "Proportion Vetoed",
    "liberal": "Liberal",
    "conservative": "Conservative",
}

# Make all normalized keys the same as the unnormalized keys
for k, v in list(LABEL_TO_FULL_NAME.items()):
    LABEL_TO_FULL_NAME[k + "_normalized"] = v


ENV_NAMES = ["Therapy-Talk", "Booking-Assistance", "Action-Advice", "Political-Questions"]


class MatchDict:
    """If the key is not found in the dict, it will look for a the key which matches a part of the key requested."""

    def __init__(self, prefix_dict, default=None):
        self.prefix_dict = prefix_dict
        self.default = default

    def __getitem__(self, key):
        for prefix, value in self.prefix_dict.items():
            if prefix in key:
                return value
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default if default is not None else self.default


def load_retro_eval_df(run, gpt=False, max_iter=None, verbose=False):
    df = load_pickle(f"data_for_figures/{run}.pkl" if not gpt else f"data_for_figures/{run}_gpt.pkl")
    if max_iter:
        df = df.query(f"iteration_number <= {max_iter}")

    # TODO: This stuff should be in the retro evaluator
    if "veto" in run.lower():
        # Compute propotion of vetoed trajectories per iteration
        if "5_veto" in run:
            assert all(df["traj_infl"].between(1, 5))
            assert all(df["traj_infl_normalized"].between(0, 1))
        else:
            assert all(df["traj_infl"].between(0, 1.01))
            assert all(df["traj_infl_normalized"].between(0, 1.01))
            df["veto_binary"] = df["traj_infl"] >= 0.5
            df["veto_binary_normalized"] = df["traj_infl"] >= 0.5

    # Get df with best iteration (and first iteration)
    best_iteration_rew = -1000
    best_iteration = None
    iteration_numbers = df["iteration_number"].unique()
    for iteration_number in iteration_numbers:
        iteration_rew_mean = df.query(f"iteration_number == {iteration_number}")["traj_rew"].mean()
        if iteration_rew_mean > best_iteration_rew:
            best_iteration_rew = iteration_rew_mean
            best_iteration = iteration_number
    first_best_iter_df = df.query(f"iteration_number == 0 or iteration_number == {best_iteration}")
    if verbose:
        print(f"best_iteration_rew: {best_iteration_rew}, best_iteration: {best_iteration}")

    return df, first_best_iter_df


def get_run_data(
    category,
    max_iter_per_run_dict={},
    titles_dict={},
    gpt=True,
    less_metrics=True,
    vuln_normal_split=False,
    verbose=False,
):
    runs = RUN_CATEGORIES[category]
    run_data = []
    max_reward_run_data = []
    for run in runs:
        run_metrics = get_metrics_to_plot(run, normalized=True, less_metrics=less_metrics)
        df, first_best_iter_df = load_retro_eval_df(
            run, gpt=gpt, max_iter=max_iter_per_run_dict.get(run, 1000), verbose=verbose
        )

        # Populate run_data
        title = titles_dict.get(run)
        if not vuln_normal_split:
            run_data.append({"df": df, "metrics": run_metrics, "title": title})
            max_reward_run_data.append({"df": first_best_iter_df, "metrics": run_metrics, "title": title})
        else:
            # Special case for vulnerable/normal split for mixed plot
            vuln_df = df.query("env_name.str.contains('vuln_')")
            normal_df = df.query("env_name.str.contains('normal_')")

            run_data.append(
                {
                    "top": {"df": vuln_df, "metrics": run_metrics, "run_name": title},
                    "bottom": {"df": normal_df, "metrics": run_metrics, "run_name": title},
                }
            )

            first_best_iter_vuln_df = first_best_iter_df.query("env_name.str.contains('vuln_')")
            first_best_iter_normal_df = first_best_iter_df.query("env_name.str.contains('normal_')")
            max_reward_run_data.append(
                {
                    "top": {"df": first_best_iter_vuln_df, "metrics": run_metrics, "run_name": title},
                    "bottom": {"df": first_best_iter_normal_df, "metrics": run_metrics, "run_name": title},
                }
            )
    return run_data, max_reward_run_data


def setup_plot_style(palette="deep"):
    # Use a widely available, professional-looking font
    plt.rcParams["font.family"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette(palette)

    # Improve grid appearance
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.7


def create_figure_and_axis(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    return fig, ax


def customize_axis(ax, xlabel, ylabel, normalized, title=None):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(width=0.5)

    if normalized:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0.9, 10.3)

    sns.despine(left=False, bottom=False)

    if title:
        ax.set_title(title, fontweight="bold", fontsize=16, pad=20)


def add_legend(ax, title="Metrics"):
    legend = ax.legend(
        title=title,
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    legend.get_frame().set_linewidth(0.5)


def save_and_show_plot(fig, run_name, plot_name):
    plot_dir = Path("figures") / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_name
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


def set_integer_x_ticks(ax, df):
    x_min, x_max = df["iteration_number"].min(), df["iteration_number"].max()
    tick_range = x_max - x_min
    tick_step = max(1, tick_range // 4)  # Ensure step is at least 1
    x_ticks = np.arange(x_min, x_max + tick_step, tick_step, dtype=int)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_xlim(x_min, x_max)  # Ensure the full range is shown


def plot_metric_evolution_per_env(df, metrics, run_name, env_name, ax=None, label_with_env=False, return_lines=False):
    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        env_data = df[(df["env_name"] == env_name) & (df["iteration_number"] == iteration)]
        for metric in metrics:
            mean, stderr = mean_and_stderr(env_data[metric])
            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    if ax is None:
        fig, ax = create_figure_and_axis()
    else:
        fig = ax.figure

    lines = []
    labels = []
    for metric in metrics:
        if label_with_env:
            label = env_name
        else:
            label = LABEL_TO_FULL_NAME.get(metric, metric)
        if return_lines:
            # Use Matplotlib's plot function for consistency with V2
            line = ax.plot(
                iterations,
                metric_data[metric]["mean"],
                label=label,
                linewidth=2.5,
                marker="o",
                markersize=6,
            )[0]
        else:
            # Use Seaborn's lineplot for backward compatibility
            line = sns.lineplot(
                x=iterations,
                y=metric_data[metric]["mean"],
                label=label,
                ax=ax,
                linewidth=2.5,
                marker="o",
                markersize=6,
            )
        lines.append(line)
        labels.append(label)
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    if not return_lines:
        set_integer_x_ticks(ax, df[df["env_name"] == env_name])
        title = f"Evolution of Metrics - {run_name}\n{env_name}" if ax is None else None
        customize_axis(
            ax,
            "Iteration",
            "Mean Metric Value",
            normalized=metrics[0].endswith("_normalized"),
            title=title,
        )
        add_legend(ax)
        plt.tight_layout()

        if ax is None:
            save_and_show_plot(fig, run_name, f"{env_name}_metric_evolution_plot.png")

        return fig, ax
    else:
        # Minimal customization for V2 functionality
        customize_axis(ax, "Iteration", "Avg Reward", normalized=metrics[0].endswith("_normalized"))
        return lines, labels


def plot_all_environments_subplots(df, metrics, run_name):
    env_names = df.env_name.unique()
    n_envs = len(env_names)

    n_cols = 3
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5 * n_rows), dpi=300)
    fig.suptitle(f"Evolution of Metrics for All Environments - {run_name}", fontsize=18, fontweight="bold")

    for idx, env_name in enumerate(env_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # type: ignore

        _, ax = plot_metric_evolution_per_env(df=df, metrics=metrics, run_name=run_name, env_name=env_name, ax=ax)  # type: ignore
        ax.set_title(f"Environment: {env_name}", fontsize=14, fontweight="bold")  # type: ignore

        set_integer_x_ticks(ax, df[df["env_name"] == env_name])

    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])  # type: ignore

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    # Ensure white background for all subplots
    fig.patch.set_facecolor("white")
    for ax in axes.flat:  # type: ignore
        ax.set_facecolor("white")

    plot_dir = PROJECT_DATA / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "all_environments_metric_evolution_subplots.png"
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"All environments metric evolution subplots saved to: {plot_path}")


def plot_split_env_subplots(df, metrics, run_name, left_envs):
    setup_plot_style()

    all_envs = df.env_name.unique()
    right_envs = [env for env in all_envs if env not in left_envs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)  # type: ignore

    def format_label(label):
        # Remove 'vuln_' prefix, capitalize first letter
        formatted = label.replace("vuln_", "").capitalize()
        # Handle hyphenated words
        if "-" in formatted:
            parts = formatted.split("-")
            formatted = parts[0] + " " + "".join(part.capitalize() for part in parts[1:])
        return formatted.replace("Implusive", "Impulsive")

    def plot_and_format_legend(ax, envs, title, idx):
        lines = []
        labels = []
        for env_name in envs:
            line, label = plot_metric_evolution_per_env(
                df=df,
                metrics=metrics,
                run_name=run_name,
                env_name=env_name,
                ax=ax,
                label_with_env=True,
                return_lines=True,
            )
            lines.extend(line)  # type: ignore
            labels.extend(label)  # type: ignore

        ax.set_title(title, fontsize=26, fontweight="bold")

        # Format environment names in the legend
        formatted_labels = [format_label(label) for label in labels]

        # Create legend for this subplot at the bottom
        ax.legend(
            lines,
            formatted_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fontsize=12,
        )

        # Increase font size for axis labels and ticks
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlabel("Iteration", fontsize=26, fontweight="bold")

        if idx == 0:
            ax.set_ylabel("Avg. Reward", fontsize=26, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        return lines, formatted_labels

    # Left subplot
    left_lines, left_labels = plot_and_format_legend(ax1, left_envs, "Training Environments", 0)

    # Right subplot
    right_lines, right_labels = plot_and_format_legend(ax2, right_envs, "Other Environments", 1)

    plt.tight_layout()

    # Adjust subplot positions to make room for the legends
    plt.subplots_adjust(bottom=0.2, wspace=0.05)

    # Ensure white background for all subplots
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    plot_name = "all_environments_metric_evolution_subplots.png"
    plot_path = PICKLE_SAVE_PATH / "../" / "figures" / plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"All environments metric evolution subplots saved to: {plot_path}")


def plot_paired_run_aggregate_metrics(
    paired_run_data: List[Dict[str, Any]], figsize: tuple = (20, 16), save_name: str = "", main_title: str = ""
) -> None:
    num_pairs = len(paired_run_data)
    fig, axes = plt.subplots(2, num_pairs, figsize=figsize, sharey=False)

    for idx, pair in enumerate(paired_run_data):
        for row, data in enumerate([pair["top"], pair["bottom"]]):
            df = data["df"]
            metrics = data["metrics"]
            title = pair.get("title", f"{pair['top']['run_name']}")

            ax = axes[row, idx]  # type: ignore
            lines, labels = plot_aggregate_metrics(
                df,
                metrics,
                title if row == 0 else None,
                ax=ax,
                show_legend=False,  # Don't show legend for any plot
            )

            set_integer_x_ticks(ax, df)

            # Remove x-label for top row
            if row == 0:
                ax.set_xlabel("")
                ax.xaxis.set_tick_params(labelbottom=False)

            # Remove y-label and ticks for all but the leftmost plot
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if idx == 0 and row == 0:
                # ax.yaxis.set_label_position("right")
                ax.set_ylabel("Avg. Metric Value\n(Vulnerable Users)")
            elif idx == 0 and row == 1:
                # ax.yaxis.set_label_position("right")
                ax.set_ylabel("Avg. Metric Value\n(Non-Vulnerable Users)")
        # Adjust the layout
    plt.suptitle(main_title, fontsize=20, fontweight="bold", y=1.1)
    plt.tight_layout()

    # Adjust the subplot positions to reduce vertical space and make room for the legend
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.06)  # 14)

    # Add the legend to the bottom of the figure
    legend = fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(labels), fontsize=10)
    legend.get_frame().set_alpha(0.8)

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        print(f"Paired run aggregate metrics plot saved to: {save_name}")
    plt.show()


def plot_multiple_run_aggregate_metrics(
    run_data: List[Dict[str, Any]],
    figsize: tuple = (20, 5.5),
    save_name: str = "",
    main_title: str = "",
    category_name: str = "",
) -> None:
    """
    Create multiple side-by-side plots, each showing aggregate metrics for a specific run.

    Args:
    run_data (List[Dict]): A list of dictionaries, each containing:
        - 'df' (pd.DataFrame): The DataFrame for the run
        - 'metrics' (List[str]): List of metrics to plot for this run
        - 'run_name' (str): Name of the run
        - 'title' (Optional[str]): Custom title for the plot, if any
    figsize (tuple): Figure size for the entire plot
    shared_y_axis (bool): Whether to use a shared y-axis across all subplots

    Returns:
    None: Displays and saves the plot
    """
    num_runs = len(run_data)
    fig, axes = plt.subplots(1, num_runs, figsize=figsize, squeeze=False)
    axes = axes.flatten()  # Flatten axes array to handle both single and multiple subplots consistently

    for idx, run_info in enumerate(run_data):

        df = run_info["df"]

        metrics = run_info["metrics"]
        title = run_info["title"] if run_info["title"] is not None else ENV_NAMES[idx]

        # Call the existing plot_aggregate_metrics function
        _, _ = plot_aggregate_metrics(df, metrics, title, ax=axes[idx])

        # Remove y-label and ticks for all but the leftmost plot
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].tick_params(axis="y", which="both", left=False, labelleft=False)

        # Move legend to bottom of the plot
        axes[idx].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Reduce space between subplots
    plt.subplots_adjust(wspace=0)

    # Adjust layout to prevent clipping of titles and labels

    plt.suptitle(main_title, fontsize=20, fontweight="bold")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        print(f"Multiple run aggregate metrics plot saved to: {save_name}")
    plt.show()


def plot_aggregate_metrics(df, metrics, title=None, ax=None, show_legend=True):

    if ax is None:
        _, ax = create_figure_and_axis(figsize=(12, 7))

    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        iteration_data = df[df["iteration_number"] == iteration]
        for metric in metrics:
            #
            mean, stderr = mean_and_stderr(iteration_data[metric])

            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    lines = []
    labels = []
    lines = []
    labels = []
    for metric in metrics:
        # [0] to get the Line2D object
        line = ax.plot(
            iterations,
            metric_data[metric]["mean"],
            label=LABEL_TO_FULL_NAME.get(metric, metric),
            linewidth=2.5,
            marker="o",
            markersize=5,
            markeredgecolor="white",  # Add white edge to markers
            markeredgewidth=0.7,  # Set the width of the marker edge
        )[0]
        lines.append(line)
        labels.append(LABEL_TO_FULL_NAME.get(metric, metric))
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    set_integer_x_ticks(ax, df)

    customize_axis(ax, "Iteration", "Avg. Metric Value", title=title, normalized=metrics[0].endswith("_normalized"))
    if show_legend:
        add_legend(ax)
    else:
        ax.get_legend().remove() if ax.get_legend() else None
    ax.set_facecolor("white")
    return lines, labels


def get_extreme_entries(results_dfs: List[pd.DataFrame], metric: str, n: int = 5) -> Dict[str, pd.DataFrame]:
    # Concatenate all DataFrames
    all_results = pd.concat(results_dfs, ignore_index=True)

    # Sort the DataFrame by the specified metric
    sorted_df = all_results.sort_values(by=metric, ascending=False)

    # Get the highest and lowest n entries
    highest = sorted_df.head(n)
    lowest = sorted_df.tail(n)

    return {"highest": highest, "lowest": lowest}


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    formatted_output = ""
    for turn in conversation:
        role = turn["role"].capitalize()
        content = turn["content"].strip()

        formatted_output += f"{role}:\n"
        formatted_output += f"{content}\n\n"

    return formatted_output.strip()


def metrics_by_run(run):
    if "therapist" in run.lower():
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
        ]
    elif "nudging_therapist_veto" in run or "action-advice" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    elif "tickets" in run or "Tickets" in run:
        metrics = [
            "omission",
            "convincing_not_to_book",
            "implies_booking_succeeded",
            "error_mentioning",
        ]
    elif "politics" in run:
        metrics = ["political_agreement", "political_inappropriateness", "harm"]
    elif "therapist1t" in run or "therapist2t" in run or "therapist3t" in run:
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
            "gaslighting",
        ]
    elif "nudging" in run or "action-advice" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    elif "sycophancy_eval" in run:
        metrics = ["sycophancy_eval"]
    else:
        raise ValueError(f"Run {run} not recognized")
    return metrics


def get_metrics_to_plot(run, normalized=False, less_metrics=False):
    metrics = ["traj_rew"]

    if "veto" in run.lower():
        if "5_veto" in run.lower():
            metrics += ["traj_infl"]
        else:
            metrics += ["veto_binary"]

    if "nudging" in run.lower() or "action-advice" in run.lower():
        metrics.append("consumed")

    metrics += metrics_by_run(run)

    if "therapist" in run.lower() and less_metrics:
        metrics_to_skip = ["harm", "deception", "manipulation", "not_going_to_judge"]
        metrics = [m for m in metrics if m not in metrics_to_skip]
    elif ("nudging" in run.lower() or "action-advice" in run.lower()) and less_metrics:
        metrics_to_skip = ["harm", "deception"]
        metrics = [m for m in metrics if m not in metrics_to_skip]

    if normalized:
        return [m + "_normalized" for m in metrics]
    return metrics


RUN_CATEGORIES = {
    "weak": [
        "weak-therapist1t-env-09_21_084743",
        "KTO_tickets-10-01_09-06-24",
        "action-advice-09_29_150113",
        "politics-09-30_06-54-40",
    ],
    "mixed": [
        "mixed-therapist1t-env-09-27_20-29-41",
        "mixed-therapist1t-env-30p-09_24_225756",
        "mixed-therapist1t-env-20p-09_25_105101",
        "mixed-therapist1t-env-10p-09_27_180249",
    ],
    "multitimestep": [
        "weak-therapist1t-env-09_10_110023",
        "weak-therapist2t-env-09_10_213941",
        "weak-therapist3t-env-09_12_221249",
    ],
    "vetos_therapist": [
        "weak-therapist1t-env-09_21_084743",
        "GPT_Veto_Therapist-09_25_155923",
        "GPT_Const_Veto_Therapist-09_25_155915",
        "5_veto_therapist_2-09-29_12-21-54",
        "negative_veto_therapist-09_29_005739",
    ],
    "vetos_politics": [
        "politics-09-30_06-54-40",
        "gpt_veto_politics-09-30_08-12-02",
        "gpt_const_veto_politics-09_30_night",
        "5_veto_politics-09_30_011050",
        "negative_veto_politics-09_30_011044",
    ],
    "vetos_tickets": [
        "KTO_tickets-10-01_09-06-24",
        "GPT_Veto_Tickets-10-01_15-20-03",
        "GPT_Const_Veto_Tickets-10-01_16-12-56",
        "5_veto_tickets-10-01_11-37-01",
        "negative_veto_tickets-10-01_14-03-53",
    ],
    "vetos_action-advice": [
        "action-advice-09_29_150113",
        "gpt_veto_action-advice-09_29_161239",
        "gpt_const_veto_action-advice-09-30_12-12-48",
        "5_veto_action-advice-09-30_12-52-24",
        "negative_veto_action-advice-09_29_161250",
    ],
    "veto_normal": [
        "GPT_Veto_Therapist-09_25_155923",
        "GPT_Veto_Tickets-10-01_15-20-03",
        "gpt_veto_action-advice-09_29_161239",
        "gpt_veto_politics-09-30_08-12-02",
    ],
    "veto_const": [
        "GPT_Const_Veto_Therapist-09_25_155915",
        "GPT_Const_Veto_Tickets-10-01_16-12-56",
        "gpt_const_veto_action-advice-09-30_12-12-48",
        "gpt_const_veto_politics-09_30_night",
    ],
    "veto_5_point": [
        "5_veto_therapist_2-09-29_12-21-54",
        "5_veto_tickets-10-01_11-37-01",
        "5_veto_action-advice-09-30_12-52-24",
        "5_veto_politics-09_30_011050",  #
    ],
    "veto_negative": [
        "negative_veto_therapist-09_29_005739",
        "negative_veto_tickets-10-01_14-03-53",
        "negative_veto_action-advice-09_29_161250",
        "negative_veto_politics-09_30_011044",
    ],
    "gemma-2-2B": [
        "gemma_2_therapist-09_25_155640",
        "gemma_2_tickets-10-01_09-54-41",
        "gemma_2_action-advice-09_29_150133",
        "gemma_2_politics-09_30_011057",
    ],
    "gemma-2-9B": [
        "gemma_9_therapist-09_25_155621",
        "gemma_9_tickets-10-01_10-31-59",
        "gemma_9_action-advice-09_29_150140",
        "gemma_9_politics-09_30_011103",
    ],
    "gemma-2-27B": [
        "gemma_27_therapist-09_26_121341",
        "gemma_27_tickets-10-01_14-51-27",
        "gemma_27_action-advice-09_29_150240",
        "gemma_27_politics-09_30_011112",
    ],
    "gemma-therapist-veto2B": [
        "therapist_a2_v2-09_27_065916",
        "therapist_a2_v9-09_27_080941",
        "therapist_a2_v27-09_28_094053",
    ],
    "gemma-therapist-veto9B": [
        "therapist_a9_v2-09_27_081459",
        "therapist_a9_v9-09_27_075431",
        "therapist_a9_v27-09_28_115107",
    ],
    "gemma-therapist-veto27B": [
        "therapist_a27_v2-09_28_094118",
        "therapist_a27_v9-09_28_094112",
        "therapist_a27_v27-09_28_094106",
    ],
    "HH-therapist": [
        "weak-therapist1t-env-09_21_084743",
        "weak_therapist1t_env_HH_25p-09-26_02-08-57",
        "weak_therapist1t_env_HH_50p-09-26_02-05-53",
        "weak_therapist1t_env_HH_75p-09-26_20-09-05",
    ],
    "HH-tickets": [
        "KTO_tickets-10-01_09-06-24",
        "tickets_mixed_HH_25p-09_27_162249",
        "tickets_mixed_HH_50p-09-27_21-08-15",
        "tickets_mixed_HH_75p-09-27_23-09-33",
    ],
    "testing": ["KTO_tickets-10-01_09-06-24", "gemma_2_tickets-10-01_09-54-41", "5_veto_tickets-10-01_11-37-01"],
}
RUNS_FLATTENED = [run for category in RUN_CATEGORIES.values() for run in category]
setup_plot_style()
