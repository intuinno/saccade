import gymnasium
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import vertebrate_env
import einops
import numpy as np


class MMJCENV(gymnasium.Env):
    def __init__(self, obs_key='image', act_key='action', seed=0):
        self._env = gymnasium.make("vertebrate_env/MMJCENV-v0", render_mode="rgb_array")
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        self.seed = seed
        
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces
        else:
            spaces = {self._obs_key: self._env.observation_space}
        
        return gymnasium.spaces.Dict(
            {
                **spaces,
               "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.bool_),
                "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.bool_),
                "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.bool_),
            }
        )

    @property
    def action_space(self):
        gymnasium_space = self._env.action_space
        return gymnasium_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["is_first"] = False
        obs["is_last"] = terminated or truncated
        obs["is_terminal"] = terminated
        return obs, reward, terminated, truncated, info

    def reset(self, seed=42, **kwargs):
        obs, info = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs, info