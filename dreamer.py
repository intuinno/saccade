from datetime import datetime
import matplotlib.pyplot as plt
import pytz
import argparse
import functools
import os
import pathlib
import sys
from envs.vec_sac_env import VecSaccadeEnvAdapter
import torchvision

os.environ["MUJOCO_GL"] = "glfw"

import numpy as np
from ruamel.yaml import YAML


sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


def append_buffer(buffer, item):
    for key, val in item.items():
        if key not in buffer:
            if isinstance(val, dict):
                buffer[key] = {}
            else:
                buffer[key] = []
        if isinstance(val, dict):
            append_buffer(buffer[key], val)
        else:
            buffer[key].append(val)


def build_batch(buffer):
    for key, val in buffer.items():
        if isinstance(val, dict):
            buffer[key] = build_batch(buffer[key])
        else:
            buffer[key] = torch.stack(buffer[key])
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    buffer = {k: swap(v) for k, v in buffer.items()}
    return buffer


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, envs):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_eval = tools.Every(config.eval_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self.sct_history = {}
        self._update_count = 0
        self._dataset = dataset
        self._envs = envs
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def get_batch(self, training=True):
        obs = self._envs.reset()
        buffer = {}
        state = None

        for i in range(self._config.batch_length):
            policy_output, state = self._policy(obs, state, training)
            obs, reward, done, info = self._envs.step(policy_output["action"])
            append_buffer(
                buffer,
                {
                    "central": obs["central"],
                    "peripheral": obs["peripheral"],
                    "is_first": obs["is_first"],
                    "is_last": obs["is_last"],
                    "is_terminal": obs["is_terminal"],
                    "GT": obs["GT"],
                    "reward": reward,
                    "discount": 1.0 - done,
                    "action": policy_output["action"],
                    "logprob": policy_output["logprob"],
                },
            )
        batch = build_batch(buffer)
        return batch

    def get_init_wm_state(self):
        init_state = self._wm.dynamics.initial(self._config.envs)
        return init_state

    def saccade_evaluation(self):
        sct_buf = []
        epochs = self._config.total_eval_batch_size // self._config.eval_batch_size
        for i in range(epochs):
            openl, sct = self._wm.saccade_video_pred(self.get_batch(training=False))
            sct_buf.append(sct)

        sct = torch.stack(sct_buf)
        sct_std = torch.std(sct, 0).cpu()
        sct = torch.mean(sct, 0).cpu()
        mse = torch.mean(sct)
        xs = range(len(sct))
        fig = plt.figure()
        plt.plot(sct)
        plt.fill_between(xs, sct + sct_std, sct - sct_std, facecolor="blue", alpha=0.5)
        plt.title(f"Scene Convergence Time at {self._step}")
        self.sct_history[self._step] = sct
        sct_history_fig = plt.figure()
        for name, value in self.sct_history.items():
            plt.plot(xs, value, label=name)
        plt.legend()
        plt.title("Scene Convergence Times")

        self._logger.video("train_openl", to_np(openl))
        self._logger.scalar("Sac_MSE", mse)
        self._logger.add_figure("SCT", fig)
        self._logger.add_figure("SCT_History", sct_history_fig)
        self._logger.save_dict("sct_history", self.sct_history)

    def estimate_state(self, state, action, obs):
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        post, prior = self._wm.dynamics.obs_step(state, action, embed, obs["is_first"])
        post["feat"] = self._wm.dynamics.get_feat(post)
        prior["feat"] = self._wm.dynamics.get_feat(prior)
        return post, prior

    def calculate_reward(self, batch, method="prediction"):
        recon = self._wm.heads["decoder"](batch["prior"]["feat"])["central"].mode()
        reward = 0.5 * (recon - batch["obs"]["central"]) ** 2
        return torch.mean(reward, 2)

    def intrinsic_reward(self, obs, state):
        latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        post, prior = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        feat = self._wm.dynamics.get_feat(prior)
        recon = self._wm.heads["decoder"](feat)["central"].mode()
        reward = 0.5 * (recon - obs["central"]) ** 2
        reward = torch.mean(reward, dim=1)
        return reward.detach().numpy()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
            if self._should_eval(step):
                self.saccade_evaluation()
            self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            # raise NotImplementedError(self._policy)
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "saccade":
        import envs.sac_env as sac_env

        mnist = torchvision.datasets.MNIST("datasets", download=True)
        images = mnist.data.numpy()[:1000]
        env = sac_env.SaccadeEnvAdapter(images, config)

        env = wrappers.OneHotAction(env)

    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    rootdir = pathlib.Path(sys.argv[0]).parent
    tz = pytz.timezone("US/Pacific")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    exp_name = config.exp_name + "_"
    exp_name += date_time
    logdir = rootdir / config.logdir / exp_name
    config.traindir = config.traindir or logdir / "train_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
    vec_envs = VecSaccadeEnvAdapter(config.eval_batch_size, config)
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # Save experiment hyperparameters
    config_backup = argparse.Namespace(**vars(config))
    config_backup.traindir = str(config.traindir)
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(logdir / "config.yml", "w") as f:
        yaml.dump(vars(config_backup), f)

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        vec_envs,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.load_model != "":
        checkpoint = torch.load(config.load_model)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            intrinsic_reward=config.intrinsic_reward,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = YAML(typ="safe").load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
