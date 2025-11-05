import datetime
import gymnasium as gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment before stepping."
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            # In gymnasium, a time limit typically results in truncation
            truncated = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) # This will now call gymnasium.Wrapper.__init__
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        obs, reward, terminated, truncated, info = self.env.step(original)
        return obs, reward, terminated, truncated, info


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be Discrete for OneHotAction wrapper."
        super().__init__(env) # This will now call gymnasium.Wrapper.__init__
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32) # gymnasium.spaces.Box
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        obs, reward, terminated, truncated, info = self.env.step(index)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) # This will now call gymnasium.Wrapper.__init__
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box( # gymnasium.spaces.Box
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces) # gymnasium.spaces.Dict

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs, info


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env) # This will now call gymnasium.Wrapper.__init__
        self._key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action[self._key])
        return obs, reward, terminated, truncated, info


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self, **kwargs):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        obs, info = self.env.reset(**kwargs)
        return obs, info
