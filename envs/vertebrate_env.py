from typing import cast
import gymnasium
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import vertebrate_env
import einops
import numpy as np
import os
import gym


def stack_camera_obs(obs):
    camera = obs["egocentric_camera"]
    camera = einops.rearrange(camera, "(k1  k2) h w c -> (k1 h) (k2 w) c", k1=4)
    obs["egocentric_camera"] = camera
    return obs


def last_camera_obs(obs):
    camera = obs["egocentric_camera"]
    camera = camera[-1]
    obs["image"] = camera
    del obs["egocentric_camera"]
    return obs


class VertebrateEnv:
    def __init__(
        self,
        obs_key="egocentric_image",
        act_key="action",
        seed=0,
        stack_camera=True,
        render_mode="rgb_array",
        temp_k=16,
        model_name="vertebrate_model",
    ):
        self._env = gymnasium.make(
            "vertebrate_env/VertebrateEnv-v0",
            render_mode=render_mode,
            temp_k=temp_k,
            model_name=model_name,
        )
        self._seed = seed
        if stack_camera:
            original_obs_space = self._env.observation_space
            K, W, H, C = original_obs_space["egocentric_camera"].shape
            new_space = spaces.Box(0, 255, shape=(W, H, C), dtype=np.uint8)

            # Create new observation space without 'egocentric_camera' and with 'image'
            new_obs_spaces = {}
            for key, space in original_obs_space.spaces.items():
                if key != "egocentric_camera":  # Exclude this key
                    new_obs_spaces[key] = space
            new_obs_spaces["image"] = new_space  # Add the new 'image' space

            # Create new Dict space
            new_obs_space = spaces.Dict(new_obs_spaces)
            self._env = TransformObservation(self._env, last_camera_obs, new_obs_space)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)  # Should raise AttributeError, not ValueError

    def _convert_gymnasium_space_to_gym(self, space):
        """Convert gymnasium space to gym space"""
        if isinstance(space, gymnasium.spaces.Box):
            return gym.spaces.Box(
                low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
            )
        elif isinstance(space, gymnasium.spaces.Discrete):
            return gym.spaces.Discrete(space.n)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(space.nvec)
        elif isinstance(space, gymnasium.spaces.MultiBinary):
            return gym.spaces.MultiBinary(space.n)
        elif isinstance(space, gymnasium.spaces.Dict):
            converted_spaces = {}
            for key, subspace in space.spaces.items():
                converted_spaces[key] = self._convert_gymnasium_space_to_gym(subspace)
            return gym.spaces.Dict(converted_spaces)
        elif isinstance(space, gymnasium.spaces.Tuple):
            converted_spaces = []
            for subspace in space.spaces:
                converted_spaces.append(self._convert_gymnasium_space_to_gym(subspace))
            return gym.spaces.Tuple(converted_spaces)
        else:
            raise NotImplementedError(
                f"Space conversion not implemented for {type(space)}"
            )

    @property
    def observation_space(self):
        if self._obs_is_dict:
            # Convert gymnasium spaces to gym spaces
            converted_spaces = {}
            for key, space in self._env.observation_space.spaces.items():
                converted_spaces[key] = self._convert_gymnasium_space_to_gym(space)
            spaces = converted_spaces
        else:
            spaces = {
                self._obs_key: self._convert_gymnasium_space_to_gym(
                    self._env.observation_space
                )
            }

        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        gymnasium_space = self._env.action_space
        if hasattr(gymnasium_space, "n"):  # Discrete space
            return gym.spaces.Discrete(gymnasium_space.n)
        else:
            # For other space types, you might need additional conversion logic
            raise NotImplementedError(
                f"Action space conversion not implemented for {type(gymnasium_space)}"
            )

    @property
    def act_space(self):
        """Alias for action_space to support embodied Driver interface"""
        # embodied Driver expects action space to be a Dict
        action_space = self.action_space
        if isinstance(action_space, gym.spaces.Dict):
            return action_space
        else:
            return gym.spaces.Dict({self._act_key: action_space})

    @property
    def obs_space(self):
        """Alias for observation_space to support embodied Driver interface"""
        # embodied Driver expects observation space to be a Dict (which it already is)
        return self.observation_space

    def step(self, action):
        # Handle action dictionary format from embodied Driver
        if isinstance(action, dict) and self._act_key in action:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        done = terminated or truncated
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = terminated
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset(seed=self._seed)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

    def render(self, mode="human", **kwargs):
        """Render the environment"""
        return self._env.render()

    def close(self):
        """Close the environment"""
        self._env.close()
