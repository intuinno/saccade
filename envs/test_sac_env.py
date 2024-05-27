import gym
from sac_env import SaccadeEnv
import torchvision

mnist = torchvision.datasets.MNIST("datasets", download=True)

images = mnist.data.numpy()[:1000]

env = SaccadeEnv(images, render_mode="human")
# env = SaccadeEnv(images, render_mode="rgb_array")

# env = RecordVideo(env, "videos")
observation, info = env.reset()

for i in range(300000000):
    if i % 10 == 0:
        action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        observation, info = env.reset()


env.close()
