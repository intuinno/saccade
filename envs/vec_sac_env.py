import gym
import torch
from gym import spaces
import numpy as np
import math
import pygame
import einops
from pygame.locals import *
import torchvision

torch._dynamo.config.capture_func_transforms = True


class VecSaccadeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        central_size=16,
        peri_size=8,
        num_loc_per_side=4,
        max_speed=2,
        seq_len=100,
        device="cpu",
        render_mode=None,
        num_environment=10,
        compile=True,
    ):
        self.width, self.height = 64, 64  # The size of the mmnist image
        self.window_width = 300
        self.window_height = 108
        self.mnist_width, self.mnist_height = 28, 28  # The size of the mnist patch
        self.central_size = central_size
        self.peri_size = peri_size
        self.num_loc_per_side = num_loc_per_side

        if num_loc_per_side == 4:
            self.loc_length = self.width // (num_loc_per_side)
        elif num_loc_per_side == 7:
            self.loc_length = self.width // (num_loc_per_side + 1)
        else:
            raise NotImplementedError

        self.max_speed = max_speed
        self.num_env = num_environment

        self.nums_per_image = 2

        mnist = torchvision.datasets.MNIST("datasets", download=True)
        self.mnist = torch.tensor(mnist.data.numpy(), device=device)
        self.num_patches = self.mnist.shape[0]

        self.lims = (
            self.width - self.mnist_width,
            self.height - self.mnist_height,
        )
        self.lims = torch.tensor(self.lims).to(device)
        self.device = device
        self.seq_len = seq_len

        self.observation_space = spaces.Dict(
            {
                "central": spaces.Box(
                    0, 255, shape=(self.central_size, self.central_size), dtype=np.uint8
                ),
                "peripheral": spaces.Box(
                    0, 255, shape=(self.peri_size, self.peri_size), dtype=np.uint8
                ),
                "loc": spaces.Discrete(self.num_loc_per_side**2),
            }
        )
        self.action_space = spaces.Discrete(self.num_loc_per_side**2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._vmap_build_canvas = torch.vmap(self._build_canvas)
        if compile:
            self.step = torch.compile(self.step)

    def reset(self, seed=None):
        # super().reset()
        super().reset(seed=seed)
        self._reset()
        self.observation, self.info = self._get_obsv()

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()

        return self.observation, self.info

    def _reset(self):
        direcs = math.pi * (
            torch.rand(
                (
                    self.num_env,
                    self.nums_per_image,
                ),
                device=self.device,
            )
            * 2
            - 1
        )
        self.indexes = torch.randint(
            self.num_patches,
            size=(self.num_env, self.nums_per_image),
            device=self.device,
        )
        speeds = torch.randint(
            0,
            self.max_speed,
            size=(
                self.num_env,
                self.nums_per_image,
            ),
            device=self.device,
        )
        self.velocs = torch.stack(
            (speeds * torch.cos(direcs), speeds * torch.sin(direcs)),
            dim=1,
        )
        self.loc = torch.randint(
            self.num_loc_per_side**2, (self.num_env,), device=self.device
        )
        self.positions = torch.mul(
            torch.rand(
                (
                    self.num_env,
                    self.nums_per_image,
                    2,
                ),
                device=self.device,
            ),
            self.lims,
        )
        self.step_count = 0

    def _get_obsv(self):
        canvas = torch.zeros(self.num_env, self.width, self.height).to(self.device)

        for i in range(self.nums_per_image):
            buf = [
                self._build_canvas(patch, pos)
                for patch, pos in zip(
                    self.mnist[self.indexes[:, i]], self.positions[:, i, :]
                )
            ]
            buf = torch.stack(buf)

            canvas += buf
        canvas = torch.where(canvas > 255.0, 255.0, canvas)
        self.canvas = canvas

        xs, ys = self._get_loc_coord()

        self.central_vision = torch.stack(
            [
                c[x : x + self.central_size, y : y + self.central_size]
                for c, x, y in zip(self.canvas, xs, ys)
            ]
        )

        self.peri_vision = torch.stack(
            [
                einops.reduce(
                    c,
                    "(w1 w2) (h1 h2) -> w1 h1",
                    "mean",
                    w1=self.peri_size,
                    h1=self.peri_size,
                )
                for c in self.canvas
            ]
        )

        observation = {
            "central": self.central_vision,
            "peripheral": self.peri_vision,
            # "loc": self.loc,
        }
        info = {"canvas": self.canvas, "loc": self.loc}
        return observation, info

    def _get_loc_coord(self):
        x, y = self.loc // self.num_loc_per_side, self.loc % self.num_loc_per_side
        x *= self.loc_length
        y *= self.loc_length
        return x, y

    def _build_canvas(self, patch, pos):
        x, y = pos.to(dtype=torch.int)
        x = torch.clamp(x, min=0, max=self.lims[0])
        y = torch.clamp(y, min=0, max=self.lims[1])
        canvas = torch.zeros((self.width, self.height), dtype=torch.uint8).to(
            self.device
        )
        canvas[x : x + self.mnist_width, y : y + self.mnist_height] = patch
        return canvas

    def step(self, actions):
        next_pos = self.positions + self.velocs
        self.velocs = torch.where(
            ((next_pos < -2) | (next_pos > self.lims + 1)),
            -1.0 * self.velocs,
            self.velocs,
        )
        self.positions = self.positions + self.velocs
        self.step_count += 1
        reward = 1.0

        if self.step_count > self.seq_len:
            done = True
        else:
            done = False

        self.loc = actions
        self.observation, self.info = self._get_obsv()

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, reward, done, False, self.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def get_surface(self, image):
        w, h = image.shape

        buf = np.zeros((w, h, 3), dtype=np.uint8)
        buf[:, :, 2] = buf[:, :, 1] = buf[:, :, 0] = image
        buf = einops.rearrange(buf, "h w c -> w h c")

        surf = pygame.surfarray.make_surface(buf)
        return surf

    def _render_frame(self):
        if self.window is None and self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height),
                HWSURFACE | DOUBLEBUF | RESIZABLE,
            )
            self.draw_screen = self.window.copy()
        if self.clock is None and self.render_mode in ["human", "rgb_array"]:
            self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_width, self.window_height))
        self.draw_screen.fill((100, 100, 100))
        pygame.display.flip()

        offset = 0
        for i in range(3):
            self.draw_env(i, offset)
            offset += 100

        if self.render_mode == "human":
            self.window.blit(
                pygame.transform.scale(self.draw_screen, self.window.get_rect().size),
                (0, 0),
            )
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.draw_screen)), axes=(1, 0, 2)
            )

    def draw_env(self, index, offset):
        image = self.canvas[index].numpy()
        surf = self.get_surface(image)
        self.draw_screen.blit(surf, (offset, 0))

        peri = self.peri_vision[index].numpy()
        surf = self.get_surface(peri)
        # self.draw_screen.blit(surf, (30 + self.box_side_length, 72))
        self.draw_screen.blit(
            pygame.transform.scale(surf, (self.central_size, self.central_size)),
            (30 + self.central_size + offset, 72),
        )

        central = self.central_vision[index].numpy()
        surf = self.get_surface(central)
        self.draw_screen.blit(surf, (10 + offset, 72))

        # Draw red line around loc
        top, left = self._get_loc_coord()
        top = top[index]
        left = left[index]
        pygame.draw.rect(
            self.draw_screen,
            (255, 0, 0),
            pygame.Rect(left + offset, top, self.central_size, self.central_size),
            width=1,
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


class VecSaccadeEnvAdapter:
    def __init__(self, configs):
        env = VecSaccadeEnv(
            num_loc_per_side=configs.num_loc_per_side,
            device=configs.device,
            num_environment=configs.envs,
        )
        self.num_envs = configs.envs
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")

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
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                "central": gym.spaces.Box(
                    0, 255, (np.prod(spaces["central"].shape),), dtype=np.uint8
                ),
                "peripheral": gym.spaces.Box(
                    0, 255, (np.prod(spaces["peripheral"].shape),), dtype=np.uint8
                ),
                "GT": gym.spaces.Box(0, 255, (64, 64), dtype=np.uint8),
                # "loc": spaces.Discrete(16),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        action_space = self._env.action_space
        action_space.discrete = True
        return action_space

    def step(self, action):
        if len(action.shape) > 1:
            action = torch.argmax(action, 1)
        obs, reward, done, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs = self.flatten_obs(obs)
        obs["is_first"] = torch.zeros((self.num_envs,))
        obs["is_last"] = torch.zeros((self.num_envs,))
        obs["is_terminal"] = torch.zeros((self.num_envs))
        obs["GT"] = info["canvas"]
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs = self.flatten_obs(obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs["GT"] = info["canvas"]
        return obs

    def flatten_obs(self, obs):
        obs = {k: torch.flatten(v, start_dim=1) for k, v in obs.items()}
        return obs
