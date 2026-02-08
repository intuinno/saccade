import os
import cv2
import gym
import logging
import numpy as np
import torch
from torch import nn
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


# ---------------------------------------------------------------------------
# Pretrained low-level policy models (minimal reimplementation for inference)
# ---------------------------------------------------------------------------

class _GoalConditionedNet(nn.Module):
    """Matches GoalConditionedNet from train_taxi_nav.py for weight loading.

    Weight keys: preprocess.goal_networks.{0..num_goals-1}.{0,2}.{weight,bias}
    """

    def __init__(self, proprioception_dim, num_goals=4,
                 hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()
        self.num_goals = num_goals
        self.output_dim = hidden_sizes[-1]
        self.goal_networks = nn.ModuleList([
            self._build_mlp(proprioception_dim, hidden_sizes, activation)
            for _ in range(num_goals)
        ])

    @staticmethod
    def _build_mlp(in_dim, hidden_sizes, activation):
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        return nn.Sequential(*layers)

    def forward(self, obs):
        goal_onehot = obs[:, :self.num_goals]
        proprioception = obs[:, self.num_goals:]
        heads = torch.stack(
            [net(proprioception) for net in self.goal_networks], dim=1,
        )
        output = (heads * goal_onehot.unsqueeze(-1)).sum(dim=1)
        return output


class _TaxiActor(nn.Module):
    """Matches ContinuousActorProbabilistic for weight loading.

    Weight keys: preprocess.*, mu.model.0.{weight,bias}, sigma_param
    forward() returns deterministic mu (mean action).
    """

    def __init__(self, proprioception_dim, action_dim=8, num_goals=4,
                 hidden_sizes=(256, 256)):
        super().__init__()
        self.preprocess = _GoalConditionedNet(
            proprioception_dim, num_goals, hidden_sizes, nn.Tanh,
        )
        # mu is an MLP wrapper; checkpoint keys are mu.model.0.{weight,bias}
        self.mu = nn.Sequential(
            nn.Linear(hidden_sizes[-1], action_dim),
        )
        # Not used for inference, but needed for weight loading
        self.sigma_param = nn.Parameter(torch.zeros(action_dim, 1))

    def forward(self, obs):
        """Return deterministic mean action."""
        features = self.preprocess(obs)
        return self.mu(features)


class MMJCHierNav:
    """Hierarchical navigation wrapper with continuous observations.

    A pretrained low-level 'taxi' policy handles motor control. The high-level
    DreamerV3 agent issues discrete navigation commands (forward/backward/CW/CCW)
    and observes continuous position, heading, and vision. The low-level policy
    executes each command for k steps. Reward is sparse (1 on goal, 0 otherwise)
    with a distance-based curriculum.
    """

    CURRICULUM_STAGES = [{"max_distance": d} for d in range(1, 10)] + [
        {"max_distance": float("inf")},
    ]
    ADVANCE_THRESHOLD = 0.5
    HISTORY_LENGTH = 100
    WALKABLE_CHARS = {'.', 'P', 'G'}

    def __init__(self, task, size=(64, 64), seed=0, model_path="",
                 k_steps=20, hidden_sizes=(256, 256), goal_radius=1.0,
                 render_mode=None):
        import gymnasium
        import mmjc_env

        # Create env with render_mode="human" to get 256x256 camera,
        # then override to rgb_array so render() returns frames without
        # opening a pygame window.
        self._env = gymnasium.make(
            task, targets_per_room=1, target_reward=False,
            global_observables=True, render_mode="human",
        )
        self._env.unwrapped.render_mode = render_mode
        self._size = size
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Parse maze_size from task name (e.g., "mmjc-13" → 13)
        self._maze_size = int(task.split("-")[-1])
        self._goal_radius = goal_radius
        self._k_steps = k_steps

        # Read sensor dim from base env
        self._sensor_dim = self._env.observation_space["sensors"].shape[0]

        # Load pretrained low-level policy
        num_goals = 4
        action_dim = self._env.action_space.shape[0]
        self._actor = _TaxiActor(
            self._sensor_dim, action_dim, num_goals, tuple(hidden_sizes),
        )
        self._load_pretrained(os.path.expanduser(model_path))
        self._actor.eval()

        # Curriculum state
        self._curriculum_stage = 0
        self._episode_outcomes = deque(maxlen=self.HISTORY_LENGTH)

        # Current goal and last base obs (set on each reset)
        self._goal_pos = None
        self._last_base_obs = None
        self._last_info = None
        self._trajectory = []
        # Optional callback invoked after each low-level step: fn() -> None
        self.substep_callback = None

    def _load_pretrained(self, model_path):
        if not os.path.exists(model_path):
            logger.warning(f"Pretrained model not found: {model_path}")
            return
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # Extract actor weights: strip "policy.actor." prefix
        actor_state = {}
        for k, v in state_dict.items():
            if k.startswith("policy.actor."):
                new_key = k.replace("policy.actor.", "")
                actor_state[new_key] = v

        # Remap mu keys: checkpoint has "mu.model.0.*" but our mu is Sequential[0]
        remapped = {}
        for k, v in actor_state.items():
            if k.startswith("mu.model.0."):
                new_key = k.replace("mu.model.0.", "mu.0.")
                remapped[new_key] = v
            elif k.startswith("preprocess."):
                remapped[k] = v
            elif k == "sigma_param":
                remapped[k] = v

        self._actor.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded pretrained taxi policy from {model_path}")

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
        spaces["current_position"] = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        spaces["target_position"] = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        spaces["heading"] = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    def _normalize_pos(self, pos):
        """Map grid position [0, maze_size-1] → [-1, 1]."""
        return (np.asarray(pos, dtype=np.float32) / (self._maze_size - 1)) * 2 - 1

    def _build_obs(self, base_obs, info, is_first, is_last, is_terminal):
        agent_pos = info["agent_pos"]
        agent_dir = info["agent_dir"]
        norm = np.linalg.norm(agent_dir)
        heading = (agent_dir / norm).astype(np.float32) if norm > 0 else np.zeros(2, dtype=np.float32)
        return {
            "current_position": self._normalize_pos(agent_pos),
            "target_position": self._normalize_pos(self._goal_pos),
            "heading": heading,
            "image": base_obs["image"],
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def _get_floor_cells(self, maze_map):
        floor_cells = []
        rows, cols = maze_map.shape
        for r in range(rows):
            for c in range(cols):
                if maze_map[r, c] in self.WALKABLE_CHARS:
                    gx = c - 1
                    gy = r - 1
                    floor_cells.append([float(gx), float(gy)])
        return np.array(floor_cells, dtype=np.float32)

    def _select_goal(self, agent_pos, floor_cells):
        max_dist = self.CURRICULUM_STAGES[self._curriculum_stage]["max_distance"]
        # Compare distances using cell centers
        cell_centers = floor_cells + 0.5
        distances = np.linalg.norm(cell_centers - agent_pos, axis=1)
        valid_mask = (distances >= 1.0) & (distances <= max_dist)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            valid_mask = distances >= 1.0
            valid_indices = np.where(valid_mask)[0]
        chosen_idx = self._rng.choice(valid_indices)
        # Return cell center, not corner
        return cell_centers[chosen_idx]

    def _maybe_advance_curriculum(self):
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
        self._last_base_obs = base_obs
        self._last_info = info
        self._trajectory = [(agent_pos[0], agent_pos[1])]
        obs = self._build_obs(base_obs, info, True, False, False)
        obs["log_curriculum_stage"] = 0.0
        return obs

    def step(self, action):
        # Convert discrete action to 4D one-hot goal vector
        # 0=forward, 1=backward, 2=rotate_CW, 3=rotate_CCW
        goal = np.zeros(4, dtype=np.float32)
        goal[action] = 1.0

        terminated = False
        truncated = False
        reached_goal = False

        for _ in range(self._k_steps):
            if self._last_base_obs is None:
                break

            proprioception = self._last_base_obs["sensors"]
            obs_in = np.concatenate([goal, proprioception])
            obs_tensor = torch.tensor(obs_in, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                mu = self._actor(obs_tensor)
                motor_action = mu.numpy()[0]
                motor_action = np.clip(motor_action, -1.0, 1.0)

            base_obs, _, base_term, base_trunc, info = self._env.step(motor_action)
            self._last_base_obs = base_obs
            self._last_info = info

            agent_pos = info["agent_pos"]
            self._trajectory.append((agent_pos[0], agent_pos[1]))

            if self.substep_callback is not None:
                self.substep_callback()

            # Check if goal reached
            dist_to_goal = np.linalg.norm(agent_pos - self._goal_pos)
            reached_goal = dist_to_goal < self._goal_radius

            if base_term or base_trunc or reached_goal:
                terminated = base_term or reached_goal
                truncated = base_trunc and not reached_goal
                break

        if reached_goal:
            reward = 1.0
        else:
            reward = 0.0

        done = terminated or truncated

        if done:
            self._episode_outcomes.append(reached_goal)
            self._maybe_advance_curriculum()

        obs = self._build_obs(
            self._last_base_obs, self._last_info, False, done, terminated,
        )
        # log_ keys are summed over the episode by the training loop;
        # only set on the terminal step so the sum equals the value.
        obs["log_curriculum_stage"] = float(self._curriculum_stage + 1) if done else 0.0
        info["curriculum_stage"] = self._curriculum_stage + 1
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        frame = self._env.render()
        if frame is None or self._goal_pos is None:
            return frame

        # Draw goal marker on the maze-map panel (3rd panel).
        cam_res = self._env.unwrapped.camera_resolution
        maze_map = self._last_info.get("maze_map") if self._last_info else None
        if maze_map is None:
            return frame

        frame = frame.copy()
        height = maze_map.shape[0]
        scale = cam_res / height
        offset_x = 2 * cam_res

        # Draw trajectory as a polyline on the 3rd panel (maze map).
        if len(self._trajectory) > 1:
            pts = []
            for px, py in self._trajectory:
                ix = int((px + 1) * scale) + offset_x
                iy = int((height - 1 - py) * scale)
                pts.append((ix, iy))
            cv2.polylines(
                frame, [np.array(pts, dtype=np.int32)],
                isClosed=False, color=(180, 180, 180), thickness=1,
            )

        # Draw goal marker (green circle). goal_pos is already at cell center.
        gx = int((self._goal_pos[0] + 1) * scale)
        gy = int((height - 1 - self._goal_pos[1]) * scale)
        radius = max(3, int(scale * 0.45))
        cv2.circle(frame, (offset_x + gx, gy), radius, (0, 255, 0), -1)
        cv2.circle(frame, (offset_x + gx, gy), radius, (0, 0, 0), 1)

        return frame

    def close(self):
        self._env.close()
