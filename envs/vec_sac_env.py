import torch
import numpy as np
import math
import einops
import torchvision
from torch.nn.functional import one_hot


class VecSaccadeEnv:

    def __init__(
        self,
        central_size=16,
        peri_size=8,
        num_loc_per_side=4,
        max_speed=2,
        seq_len=1000,
        device="cpu",
        batch_size=10,
    ):
        self.width, self.height = 64, 64  # The size of the mmnist image
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

        self.num_loc = num_loc_per_side**2

        self.max_speed = max_speed
        self.num_env = batch_size

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

    def reset(self):
        self._reset()
        self.observation = self._get_obsv()

        return self.observation

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
        self.loc = torch.randint(self.num_loc, (self.num_env,), device=self.device)
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

        self.peri_vision = torch.flatten(self.peri_vision, start_dim=1)
        self.central_vision = torch.flatten(self.central_vision, start_dim=1)

        observation = {
            "central": self.central_vision,
            "peripheral": self.peri_vision,
            "loc": one_hot(self.loc, num_classes=self.num_loc),
            "GT": self.canvas,
        }
        return observation

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
        reward = torch.zeros((self.num_env), dtype=torch.float32).to(self.device)

        # if self.step_count > self.seq_len:
        #     done = True
        # else:
        done = torch.zeros((self.num_env), dtype=torch.int32).to(self.device)

        self.loc = torch.argmax(actions, dim=1)
        self.observation = self._get_obsv()

        return self.observation, reward, done

    def flatten_obs(self, obs):
        obs = {k: torch.flatten(v, start_dim=1) for k, v in obs.items()}
        return obs
