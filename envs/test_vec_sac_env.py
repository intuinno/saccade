import gym
from gym.wrappers import RecordVideo
from vec_sac_env import VecSaccadeEnv
import torchvision
import torch

num_env = 7
env = VecSaccadeEnv(render_mode="human", num_environment=num_env)
# env = SaccadeEnv(images, render_mode="rgb_array")

# env = RecordVideo(env, "videos")
observation, info = env.reset()

for i in range(3000):
    if i % 1 == 0:
        actions = torch.randint(16, (num_env,))
    observation, reward, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        observation, info = env.reset()


env.close()
