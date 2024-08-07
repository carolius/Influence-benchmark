import copy
from multiprocessing import Queue
from typing import Dict, List, Tuple

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_character import VectorizedCharacter
from influence_benchmark.vectorized_environment.vectorized_influence_detector_model import (
    VectorizedInfluenceDetectorModel,
)
from influence_benchmark.vectorized_environment.vectorized_preference_model import VectorizedPreferenceModel
from influence_benchmark.vectorized_environment.vectorized_transition_model import VectorizedTransitionModel


class VectorizedEnvironment:
    """
    A class representing a vectorized environment for running multiple environments in parallel.
    """

    def __init__(self, backend: Backend, max_envs: int, shared_queue: Queue, progress):
        """
        Initialize the VecEnv with multiple environment configurations and a backend.

        Args:
            env_configs (List[Dict]): A list of environment configurations.
            backend (Backend): The backend to use for computations.
        """
        self.max_envs = max_envs
        self.backend = backend
        self.environments = {}
        self.traj_count = {}
        self.shared_queue = shared_queue
        self.progress = progress

        self.setup_models()

    def setup_models(self):
        """
        Set up the vectorized models (transition, preference, and character) for the environments.
        """

        self.vectorized_preference_model = VectorizedPreferenceModel(self.backend, self.max_envs)
        self.vectorized_influence_detector_model = VectorizedInfluenceDetectorModel(self.backend, self.max_envs)
        self.vectorized_transition_model = VectorizedTransitionModel(self.backend, self.max_envs)

        self.vectorized_character = VectorizedCharacter(self.backend, self.max_envs)
        for i in range(self.max_envs):
            models = self.shared_queue.get()
            self.environments[i] = models["environment"]
            self.vectorized_preference_model.add_model(models["preference_model"], i)
            self.vectorized_influence_detector_model.add_model(models["influence_detector_model"], i)
            self.vectorized_transition_model.add_model(models["transition_model"], i)
            self.vectorized_character.add_model(models["character"], i)
            self.traj_count[i] = 0

    def replace_environment(self, env_id: int):
        self.progress.value += 1
        new_env = self.shared_queue.get()
        if new_env is None:
            del self.environments[env_id]
            self.vectorized_preference_model.remove_model(env_id)
            self.vectorized_influence_detector_model.remove_model(env_id)
            self.vectorized_transition_model.remove_model(env_id)
            self.vectorized_character.remove_model(env_id)
            del self.traj_count[env_id]
        else:
            self.environments[env_id] = new_env["environment"]
            self.vectorized_preference_model.replace_model(new_env["preference_model"], env_id)
            self.vectorized_influence_detector_model.replace_model(new_env["influence_detector_model"], env_id)
            self.vectorized_transition_model.replace_model(new_env["transition_model"], env_id)
            self.vectorized_character.replace_model(new_env["character"], env_id)
            self.traj_count[env_id] = 0

    def get_envs(self) -> List[Environment]:
        keys = sorted(self.environments.keys())
        return [self.environments[key] for key in keys]

    def reset(self) -> List[Dict]:
        """
        Reset all environments and return their initial observations.

        Returns:
            List[Dict]: A list of initial observations for all environments.
        """
        return [env.reset() for env in self.get_envs()]

    def step_vec(self, action_n: List[str], tool_calls, tool_results) -> Tuple[List[State], List[bool]]:
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
        next_state_n = self._vectorized_step(state_n, action_n, tool_calls, tool_results)

        for env, next_state in zip(self.get_envs(), next_state_n):
            env.current_state = next_state

        done_n = [env.is_terminal(next_state) for env, next_state in zip(self.get_envs(), next_state_n)]

        return next_state_n, done_n

    def _vectorized_step(self, state_n: List[State], action_n: List[str], tool_calls, tool_results) -> List[State]:
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
        active_calls = [call for call, action in zip(tool_calls, action_n) if action is not None]
        active_results = [result for result, action in zip(tool_results, action_n) if action is not None]

        for state, action, call, result in zip(active_states, active_actions, active_calls, active_results):
            if call is not None:
                state.history.append({"role": "tool", "content": call})
                state.history.append({"role": "ipython", "content": result})
            state.history.append({"role": "agent", "content": action})

        # The transition is computed on the environment response.

        next_state_n = self.vectorized_transition_model.get_next_states(active_states, active_actions, self.get_envs())

        # The preference model and influence scores are calculated on the agent's response
        self.vectorized_preference_model.add_preferences_to_states(next_state_n)
        self.vectorized_influence_detector_model.add_influence_scores_to_states(next_state_n)
        self.vectorized_character.add_char_responses_to_states(next_state_n)

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

    def generate_trajectories(
        self,
        agent: Agent,
        n_trajs_per_initial_state: int,
    ) -> List[Dict]:
        """
        Generate trajectories for all environments using the provided agent.
        """
        env_trajectories = []
        while self.get_num_envs() > 0:

            is_done_n = self.reset_done_envs()
            for id, done in is_done_n.items():
                if done and self.get_trajectory_count(id) >= n_trajs_per_initial_state:
                    self.replace_environment(id)
            if self.get_num_envs() == 0:
                break
            observations = self.get_observation_vec()
            actions, tool_calls, tool_results = agent.get_action_vec(observations)
            next_states, _ = self.step_vec(actions, tool_calls, tool_results)

            for i, env in self.environments.items():
                env_trajectories.append(
                    {
                        "env_name": env.env_name,
                        "initial_state_id": env.config["history_id"],
                        "trajectory_id": self.get_trajectory_count(i),
                        "turn": env.current_state.turns,
                        "agent_system_prompt": agent.get_system_prompt(env.current_state),
                        "history": env.current_state.history[:-1],
                        "preferences": env.current_state.preferences,
                        "influence_scores": env.current_state.influence_scores,
                        "transition_probs": env.current_state.transition_probs,
                    }
                )
        return env_trajectories

    def get_terminal_status(self) -> List[bool]:
        """
        Get the terminal status of all environments.

        Returns:
            List[bool]: A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        return [env.is_terminal(env.current_state) for env in self.get_envs()]

    def reset_done_envs(self):
        """
        Reset all environments that have reached a terminal state.

        Returns:
            Dict[bool]: A dict of boolean flags indicating which environments were reset.
        """
        is_done_n = {}
        for env_id, env in self.environments.items():
            if env.is_terminal(env.current_state):
                env.reset()
                self.traj_count[env_id] += 1
                is_done_n[env_id] = True
            else:
                is_done_n[env_id] = False
        return is_done_n

    def get_trajectory_count(self, id: int) -> int:
        return self.traj_count[id]

    def get_observation_vec(self) -> List[State]:
        """
        Get observations from all environments.

        Returns:
            List[Dict]: A list of observations, one for each environment.
        """
        return [env.get_observation() for env in self.get_envs()]

    def get_num_envs(self) -> int:
        """
        Get the number of environments in the VecEnv.

        Returns:
            int: The number of environments.
        """
        return len(self.environments)
