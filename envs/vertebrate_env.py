from typing import cast
import gymnasium
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import vertebrate_env
import einops
import numpy as np



def stack_camera_obs(obs):
    camera = obs['egocentric_camera']
    camera = einops.rearrange(camera, "(k1  k2) h w c -> (k1 h) (k2 w) c", k1=4)
    obs['egocentric_camera'] = camera
    return obs
    
def last_camera_obs(obs):
    camera = obs['egocentric_camera']
    camera = camera[-1]
    obs['image'] = camera
    del obs['egocentric_camera']
    return obs
    

class VertebrateEnv(gymnasium.Env):
    def __init__(self, obs_key='egocentric_image', act_key='action', seed=0,  stack_camera=True,):
        self._env = gymnasium.make("vertebrate_env/VertebrateEnv-v0", render_mode="rgb_array", temp_k=16, seed=seed)
        if stack_camera:
            original_obs_space = self._env.observation_space
            K, W, H, C = original_obs_space['egocentric_camera'].shape
            new_space = spaces.Box(0, 255, shape=(W, H, C), dtype=np.uint8)
            
            # Create new observation space without 'egocentric_camera' and with 'image'
            new_obs_spaces = {}
            for key, space in original_obs_space.spaces.items():
                if key != 'egocentric_camera':  # Exclude this key
                    new_obs_spaces[key] = space
            new_obs_spaces['image'] = new_space  # Add the new 'image' space
            
            # Create new Dict space
            new_obs_space = spaces.Dict(new_obs_spaces)
            self._env = TransformObservation(self._env, last_camera_obs, new_obs_space)
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        
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