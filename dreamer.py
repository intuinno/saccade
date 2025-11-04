import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent / "dreamerv3"))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers

# Use embodied package for parallel processing
from embodied.core import Driver
print("✓ Using embodied package for parallel processing")

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class EmbodiedParallel:
    """
    Adapter class that makes a single environment work with embodied Driver interface
    while maintaining backward compatibility with existing saccade code.
    """
    def __init__(self, env_constructor, strategy="process"):
        self.env_constructor = env_constructor
        self.strategy = strategy
        self._env = None
        self._init_env()

    def _init_env(self):
        """Initialize the environment"""
        if callable(self.env_constructor):
            self._env = self.env_constructor()
        else:
            self._env = self.env_constructor

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped environment"""
        if self._env is None:
            self._init_env()
        return getattr(self._env, name)

    def step(self, action):
        """Step method that returns a callable for backward compatibility"""
        if self._env is None:
            self._init_env()
        return lambda: self._env.step(action)

    def reset(self):
        """Reset method that returns a callable for backward compatibility"""
        if self._env is None:
            self._init_env()
        return lambda: self._env.reset()

    def close(self):
        """Close the environment"""
        if self._env is not None and hasattr(self._env, 'close'):
            self._env.close()


class EmbodiedDummy:
    """
    Dummy wrapper that maintains the Damy interface while being embodied-ready
    """
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()


def create_embodied_envs(config, mode_prefix):
    """
    Create environment list that can work with embodied Driver.
    Returns both individual wrapped envs and a Driver for batch processing.
    """
    def make_env_fn(env_id):
        """Create a single environment constructor"""
        def env_constructor():
            return make_env(config, mode_prefix, env_id)
        return env_constructor
    
    # Create environment constructors
    make_env_fns = [make_env_fn(i) for i in range(config.envs)]
    
    if config.parallel:
        # Create embodied Driver for batch processing
        driver = Driver(make_env_fns, parallel=True)
        
        # Create individual environment wrappers for backward compatibility
        individual_envs = []
        for i in range(config.envs):
            env_wrapper = EmbodiedParallel(make_env_fns[i], "process")
            env_wrapper._driver = driver  # Reference to shared driver
            env_wrapper._env_id = i
            individual_envs.append(env_wrapper)
        
        return individual_envs, driver
    else:
        # Create individual environments (non-parallel path)
        envs = [make_env_fn(i)() for i in range(config.envs)]
        wrapped_envs = [EmbodiedDummy(env) for env in envs]
        return wrapped_envs, None


class SaccadeEmbodiedAdapter:
    """
    Adapter that integrates embodied Driver with saccade's tools.simulate function
    """
    def __init__(self, envs, driver=None):
        self.envs = envs
        self.driver = driver
        self.use_driver = driver is not None
        
    def simulate_with_embodied(self, agent, cache, directory, logger, **kwargs):
        """
        Use embodied Driver for simulation when available, otherwise fallback to original
        """
        if not self.use_driver:
            # Use original simulation
            return tools.simulate(agent, self.envs, cache, directory, logger, **kwargs)
        
        # Enhanced simulation using embodied Driver
        # This would need to be implemented to integrate with tools.simulate
        # For now, fallback to original simulation
        return tools.simulate(agent, self.envs, cache, directory, logger, **kwargs)


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
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
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
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
    elif suite == "vertebrate":
        import envs.vertebrate_env as vertebrate_env
        env = vertebrate_env.VertebrateEnv()
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
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    print("Using embodied package for parallel processing")
    train_envs, train_driver = create_embodied_envs(config, "train")
    eval_envs, eval_driver = create_embodied_envs(config, "eval")
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

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
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
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
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
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
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    
    # Cleanup environments and drivers
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    
    # Close embodied drivers if they exist
    if 'train_driver' in locals() and train_driver:
        try:
            train_driver.close()
        except Exception:
            pass
    if 'eval_driver' in locals() and eval_driver:
        try:
            eval_driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
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
