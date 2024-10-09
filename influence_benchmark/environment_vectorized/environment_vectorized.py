import copy
from typing import Dict, List, Optional, Tuple

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.environment_vectorized.character_vectorized import VectorizedCharacter
from influence_benchmark.environment_vectorized.influence_detector_model_vectorized import (
    VectorizedInfluenceDetectorModel,
)
from influence_benchmark.environment_vectorized.preference_model_vectorized import VectorizedPreferenceModel
from influence_benchmark.environment_vectorized.trajectory_queue import TrajectoryQueue
from influence_benchmark.environment_vectorized.transition_model_vectorized import VectorizedTransitionModel


class VectorizedEnvironment:
    """
    A class representing a vectorized environment for running multiple environments in parallel.
    """

    def __init__(
        self,
        backends: Dict[str, Backend],
        max_envs: int,
        shared_queue: TrajectoryQueue,
        progress,
        pm_length_penalty: Optional[float] = None,
    ):
        """
        Initialize the VectorizedEnvironment with multiple environment configurations and backends.

        Args:
            backends (Dict[str, Backend]): A dictionary of backends for different components.
            max_envs (int): The maximum number of environments to run in parallel.
            shared_queue (TrajectoryQueue): A shared queue for trajectory data.
            progress: A progress tracker object.
            pm_length_penalty (Optional[float]): A length penalty for the preference model. Defaults to None.
        """
        self.max_envs = max_envs
        self.backends = backends
        self.environments = {}
        self.current_traj_ids = {}
        self.shared_queue = shared_queue
        self.progress = progress
        self.pm_length_penalty = pm_length_penalty

        self.setup_models()

    def setup_models(self):
        """
        Set up the vectorized models (transition, preference, and character) for the environments.
        """
        self.preference_model_vectorized = VectorizedPreferenceModel(
            self.backends.get("env-preference", self.backends["env"]), self.max_envs, self.pm_length_penalty
        )
        self.influence_detector_model_vectorized = VectorizedInfluenceDetectorModel(
            self.backends.get("env-influence", self.backends["env"]), self.max_envs
        )
        self.transition_model_vectorized = VectorizedTransitionModel(
            self.backends.get("env-transition", self.backends["env"]), self.max_envs
        )
        self.character_vectorized = VectorizedCharacter(
            self.backends.get("env-character", self.backends["env"]), self.max_envs
        )

        for i in range(self.max_envs):
            subenv_models = self.shared_queue.get()
            if subenv_models is None:
                print("WARNING: you're using too many GPUs for the number of trajectories you're generating!")
                continue
            self.current_traj_ids[i] = subenv_models["traj_id"]
            self.environments[i] = subenv_models["environment"]
            self.preference_model_vectorized.add_model(subenv_models["preference_model"], i)
            self.influence_detector_model_vectorized.add_model(subenv_models["influence_detector_model"], i)
            self.transition_model_vectorized.add_model(subenv_models["transition_model"], i)
            self.character_vectorized.add_model(subenv_models["character"], i)

    def remove_environment(self, env_id: int):
        """
        Remove an environment and its associated models from the vectorized environment.

        Args:
            env_id (int): The ID of the environment to remove.
        """
        del self.environments[env_id]
        self.preference_model_vectorized.remove_model(env_id)
        self.influence_detector_model_vectorized.remove_model(env_id)
        self.transition_model_vectorized.remove_model(env_id)
        self.character_vectorized.remove_model(env_id)

    def env_id_to_env_position(self, env_id: int) -> int:
        """
        Get the position of an environment in the vectorized environment.

        Args:
            env_id (int): The ID of the environment.

        Returns:
            int: The position of the environment in the vectorized environment.
        """
        return sorted(self.environments.keys()).index(env_id)

    def replace_environment(self, env_id: int):
        """
        Replace an environment with a new one from the shared queue.

        Args:
            env_id (int): The ID of the environment to replace.
        """
        self.progress.value += 1
        subenv_models = self.shared_queue.get()
        if subenv_models is None:
            # This means that there are no more environments to run, so we can clean things up and clear GPU memory
            # NOTE: maybe we should remove this, as it increases chance of other people getting GPU memory and breaking our runs
            # Note that if we do remove this, it will break the logic to tell whether we're done.
            self.remove_environment(env_id)
        else:
            self.environments[env_id] = subenv_models["environment"]
            self.preference_model_vectorized.replace_model(subenv_models["preference_model"], env_id)
            self.influence_detector_model_vectorized.replace_model(subenv_models["influence_detector_model"], env_id)
            self.transition_model_vectorized.replace_model(subenv_models["transition_model"], env_id)
            self.character_vectorized.replace_model(subenv_models["character"], env_id)
            self.current_traj_ids[env_id] = subenv_models["traj_id"]

    def get_envs(self) -> List[Environment]:
        """
        Get a list of all current environments.

        Returns:
            List[Environment]: A list of Environment objects.
        """
        keys = sorted(self.environments.keys())
        return [self.environments[key] for key in keys]

    def step_vec(self, action_n: List[str]) -> Tuple[List[State], List[bool]]:
        """
        Take a step in all environments using the provided actions.

        Args:
            action_n (List[str]): A list of actions, one for each environment.

        Returns:
            Tuple[List[State], List[bool]]: A tuple containing:
                - A list of next states for all environments.
                - A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        state_n = [env.current_state for env in self.get_envs()]
        next_state_n = self._step_vectorized(state_n, action_n)

        for env, next_state in zip(self.get_envs(), next_state_n):
            env.current_state = next_state

        done_n = [env.is_terminal(next_state) for env, next_state in zip(self.get_envs(), next_state_n)]

        return next_state_n, done_n

    def _step_vectorized(self, state_n: List[State], action_n: List[str]) -> List[State]:
        """
        Perform a vectorized step for all active environments.

        Args:
            state_n (List[State]): A list of current states for all environments.
            action_n (List[str]): A list of actions for all environments.

        Returns:
            List[State]: A list of next states for all environments.
        """
        # Filter out environments that have reached a terminal state
        active_states = [copy.deepcopy(state) for state, action in zip(state_n, action_n) if action is not None]
        active_actions = [action for action in action_n if action is not None]

        for state, action in zip(active_states, active_actions):
            state.history.append({"role": "agent", "content": action})
        # The transition is computed on the environment response.

        next_state_n = self.transition_model_vectorized.get_next_states(active_states, active_actions, self.get_envs())

        # The preference model and influence scores are calculated on the agent's response
        self.preference_model_vectorized.add_preferences_to_states(next_state_n)
        self.influence_detector_model_vectorized.add_influence_scores_to_states(next_state_n)
        self.character_vectorized.add_char_responses_to_states(next_state_n)

        # Merge the active and inactive states
        merged_states = []
        active_index = 0
        for original_state, action in zip(state_n, action_n):
            if action is None:
                print("None action")
                merged_states.append(original_state)
            else:
                merged_states.append(next_state_n[active_index])
                active_index += 1

        return merged_states

    def generate_trajectories(self, agent: Agent) -> List[Dict]:
        """
        Generate trajectories for all environments using the provided agent.

        Args:
            agent (Agent): The agent to use for generating actions.

        Returns:
            List[Dict]: A list of dictionaries containing trajectory data for each environment.
        """
        env_trajectories = []
        while self.get_num_envs() > 0:
            observations = self.get_observation_vec()
            actions = agent.get_action_vec(observations)
            _ = self.step_vec(actions)

            for i, env in self.environments.items():
                env_trajectories.append(
                    {
                        "env_name": env.env_name,
                        "initial_state_id": env.history_id,
                        "trajectory_id": self.current_traj_ids[i],
                        "turn": env.current_state.turns,
                        "agent_system_prompt": agent.get_system_prompt(env.current_state),
                        "history": env.current_state.history[:-1],
                        "preferences": env.current_state.preferences,
                        "influence_scores": env.current_state.influence_scores,
                        "transition_probs": env.current_state.transition_probs,
                        "visited_states": list(env.visited_states),
                    }
                )

            is_done_n = self.get_done_envs()
            for id, done in is_done_n.items():
                if done:
                    self.replace_environment(id)

        return env_trajectories

    def get_terminal_status(self) -> List[bool]:
        """
        Get the terminal status of all environments.

        Returns:
            List[bool]: A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        return [env.is_terminal(env.current_state) for env in self.get_envs()]

    def get_done_envs(self):
        """
        Get all environments that have reached a terminal state.

        Returns:
            Dict[bool]: A dict of boolean flags indicating which environments were reset.
        """
        is_done_n = {}
        for env_id, env in self.environments.items():
            if env.is_terminal(env.current_state):
                is_done_n[env_id] = True
            else:
                is_done_n[env_id] = False
        return is_done_n

    def get_observation_vec(self) -> List[Dict]:
        """
        Get observations from all environments.

        Returns:
            List[Dict]: A list of observations, one for each environment.
        """
        return [env.get_observation() for env in self.get_envs()]

    def get_num_envs(self) -> int:
        """
        Get the number of environments in the VectorizedEnvironment.

        Returns:
            int: The number of environments.
        """
        return len(self.environments)
