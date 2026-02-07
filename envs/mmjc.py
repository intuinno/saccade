import gym
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class MMJC:
    def __init__(self, task, size=(64, 64), seed=0, render_mode=None):
        import gymnasium
        import mmjc_env

        self._env = gymnasium.make(task, targets_per_room=1, render_mode=render_mode)
        self._size = size

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        spaces = {}
        for key, space in self._env.observation_space.spaces.items():
            spaces[key] = gym.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype,
            )
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = terminated
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()


class MMJCNav:
    """Navigation wrapper with one-hot observations, sparse reward, and curriculum.

    The agent receives one-hot encoded position/heading observations and a
    sparse terminal reward of 1.0 when it reaches a randomly-chosen floor cell
    in the maze. A distance-based curriculum progressively increases goal
    distance as the agent's success rate improves.

    Goals are sampled from walkable floor cells rather than pre-placed targets,
    guaranteeing valid goals at any curriculum distance without re-resets.
    """

    CURRICULUM_STAGES = [
        {"max_distance": 3},
        {"max_distance": 5},
        {"max_distance": 8},
        {"max_distance": 11},
        {"max_distance": float("inf")},
    ]
    ADVANCE_THRESHOLD = 0.5
    HISTORY_LENGTH = 100
    WALKABLE_CHARS = {'.', 'P', 'G'}

    def __init__(self, task, size=(64, 64), seed=0, render_mode=None,
                 n_heading_bins=8, goal_radius=1.0):
        import gymnasium
        import mmjc_env

        self._env = gymnasium.make(
            task, targets_per_room=1, target_reward=False,
            global_observables=True,
        )
        self._size = size
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Parse maze_size from task name (e.g., "mmjc-13" → 13)
        self._maze_size = int(task.split("-")[-1])
        self._n_heading_bins = n_heading_bins
        self._goal_radius = goal_radius

        # Read actual sensor dim from base env observation space
        self._sensor_dim = self._env.observation_space["sensors"].shape[0]

        # Curriculum state
        self._curriculum_stage = 0
        self._episode_outcomes = deque(maxlen=self.HISTORY_LENGTH)

        # Current goal (set on each reset)
        self._goal_pos = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        spaces["goal_x"] = gym.spaces.Box(0, 1, (self._maze_size,), dtype=np.float32)
        spaces["goal_y"] = gym.spaces.Box(0, 1, (self._maze_size,), dtype=np.float32)
        spaces["position_x"] = gym.spaces.Box(0, 1, (self._maze_size,), dtype=np.float32)
        spaces["position_y"] = gym.spaces.Box(0, 1, (self._maze_size,), dtype=np.float32)
        spaces["current_heading"] = gym.spaces.Box(
            0, 1, (self._n_heading_bins,), dtype=np.float32
        )
        spaces["proprioceptors"] = gym.spaces.Box(
            -np.inf, np.inf, (self._sensor_dim,), dtype=np.float64
        )
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=np.float32,
        )

    def _pos_to_onehot(self, val):
        """Convert a grid coordinate to a one-hot vector of size maze_size."""
        idx = np.clip(int(val), 0, self._maze_size - 1)
        vec = np.zeros(self._maze_size, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _dir_to_onehot(self, agent_dir):
        """Convert a 2D direction vector to a one-hot heading vector."""
        angle = np.arctan2(agent_dir[1], agent_dir[0])  # [-pi, pi]
        angle = angle % (2 * np.pi)  # [0, 2pi)
        bin_idx = int(angle / (2 * np.pi / self._n_heading_bins))
        bin_idx = np.clip(bin_idx, 0, self._n_heading_bins - 1)
        vec = np.zeros(self._n_heading_bins, dtype=np.float32)
        vec[bin_idx] = 1.0
        return vec

    def _build_obs(self, base_obs, info, is_first, is_last, is_terminal):
        """Build the observation dict from base env obs and info."""
        obs = {}
        obs["image"] = base_obs["image"]
        obs["goal_x"] = self._pos_to_onehot(self._goal_pos[0])
        obs["goal_y"] = self._pos_to_onehot(self._goal_pos[1])
        obs["position_x"] = self._pos_to_onehot(info["agent_pos"][0])
        obs["position_y"] = self._pos_to_onehot(info["agent_pos"][1])
        obs["current_heading"] = self._dir_to_onehot(info["agent_dir"])
        obs["proprioceptors"] = base_obs["sensors"]
        obs["is_first"] = is_first
        obs["is_last"] = is_last
        obs["is_terminal"] = is_terminal
        return obs

    def _get_floor_cells(self, maze_map):
        """Extract walkable floor cell positions in grid coordinates.

        The maze_map entity_layer has an outer wall border, so map index
        (row, col) corresponds to grid coordinate (col-1, row-1).
        """
        floor_cells = []
        rows, cols = maze_map.shape
        for r in range(rows):
            for c in range(cols):
                if maze_map[r, c] in self.WALKABLE_CHARS:
                    # Convert map (row, col) to grid coords (x, y)
                    # col-1 → grid x, row-1 → grid y
                    gx = c - 1
                    gy = r - 1
                    floor_cells.append([float(gx), float(gy)])
        return np.array(floor_cells, dtype=np.float32)

    def _select_goal(self, agent_pos, floor_cells):
        """Select a random floor cell within curriculum distance as the goal."""
        max_dist = self.CURRICULUM_STAGES[self._curriculum_stage]["max_distance"]
        distances = np.linalg.norm(floor_cells - agent_pos, axis=1)
        # Exclude cells too close to agent (< 1.0) to avoid trivial goals
        valid_mask = (distances >= 1.0) & (distances <= max_dist)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            # Fallback: pick any floor cell that isn't the agent's cell
            valid_mask = distances >= 1.0
            valid_indices = np.where(valid_mask)[0]
        chosen_idx = self._rng.choice(valid_indices)
        return floor_cells[chosen_idx]

    def _maybe_advance_curriculum(self):
        """Advance curriculum stage if success rate exceeds threshold."""
        if self._curriculum_stage >= len(self.CURRICULUM_STAGES) - 1:
            return
        if len(self._episode_outcomes) < self.HISTORY_LENGTH:
            return
        success_rate = np.mean(self._episode_outcomes)
        if success_rate > self.ADVANCE_THRESHOLD:
            self._curriculum_stage += 1
            logger.info(
                f"Curriculum advanced to stage {self._curriculum_stage + 1} "
                f"(max_dist={self.CURRICULUM_STAGES[self._curriculum_stage]['max_distance']})"
            )

    def reset(self):
        base_obs, info = self._env.reset()
        agent_pos = info["agent_pos"]
        maze_map = info["maze_map"]
        floor_cells = self._get_floor_cells(maze_map)
        self._goal_pos = self._select_goal(agent_pos, floor_cells)
        return self._build_obs(base_obs, info, True, False, False)

    def step(self, action):
        base_obs, _, base_terminated, base_truncated, info = self._env.step(action)

        # Check if agent reached the goal
        agent_pos = info["agent_pos"]
        dist_to_goal = np.linalg.norm(agent_pos - self._goal_pos)
        reached_goal = dist_to_goal < self._goal_radius

        if reached_goal:
            reward = 1.0
            terminated = True
            truncated = False
        elif base_terminated or base_truncated:
            reward = 0.0
            terminated = False
            truncated = True
        else:
            reward = 0.0
            terminated = False
            truncated = False

        done = terminated or truncated

        # Track episode outcome and update curriculum when episode ends
        if done:
            self._episode_outcomes.append(reached_goal)
            self._maybe_advance_curriculum()

        obs = self._build_obs(base_obs, info, False, done, terminated)
        info["curriculum_stage"] = self._curriculum_stage + 1
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()
