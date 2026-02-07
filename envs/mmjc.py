import gym
import numpy as np


class MMJC:
    def __init__(self, task, size=(64, 64), seed=0):
        import gymnasium
        import mmjc_env

        self._env = gymnasium.make(task)
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

    def close(self):
        self._env.close()
