import numpy as np
import argparse
from ruamel.yaml import YAML
import torch
import pathlib
import sys
from tqdm import tqdm
from envs.vec_sac_env import VecSaccadeEnvAdapter
from saccade_agent import SaccadeAgent

import tools
from datetime import datetime
import pytz

to_np = lambda x: x.detach().cpu().numpy()

# torch.autograd.set_detect_anomaly(True)


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
    return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--load_model", type=str)
    args, remaining = parser.parse_known_args()
    rootdir = pathlib.Path(sys.argv[0]).parent
    yaml = YAML(typ="safe")
    configs = yaml.load((rootdir / "configs.yaml").read_text())

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
    configs = parser.parse_args(remaining)

    exp_name = configs.exp_name + "_"
    tools.set_seed_everywhere(configs.seed)
    if configs.deterministic_run:
        tools.enable_deterministic_run()

    tz = pytz.timezone("US/Pacific")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    exp_name += date_time

    # Creating model dir with experiment name.
    exp_logdir = rootdir / configs.logdir / exp_name
    print("Logdir", exp_logdir)
    exp_logdir.mkdir(parents=True, exist_ok=True)

    # Setting up environments
    envs = VecSaccadeEnvAdapter(configs)
    acts = envs.action_space
    print("Action Space", acts)
    configs.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    num_epochs = configs.total_steps // configs.envs // configs.num_steps
    configs.use_amp = True if configs.precision == 16 else False

    # Build logger
    logger = tools.Logger(exp_logdir, 0)
    metrics = {}

    # Dumping config.
    yaml.default_flow_style = False
    with open(exp_logdir / "config.yml", "w") as f:
        yaml.dump(vars(configs), f)

    agent = SaccadeAgent(
        envs.observation_space,
        envs.action_space,
        configs,
        logger,
    ).to(configs.device)

    print(f"========== Using {configs.device} device ===================")

    # Main Loop
    step = 0
    _ = envs.reset()
    init_state = agent.get_init_wm_state()
    state = init_state

    for i in tqdm(range(int(num_epochs))):
        with tools.RequiresGrad(agent):
            with torch.cuda.amp.autocast(configs.use_amp):
                buffer = {}
                for j in range(configs.num_steps):
                    action, logprob, actor_ent = agent.get_action(state)
                    obs, _, _, _ = envs.step(action)
                    post, prior = agent.estimate_state(state, action, obs)
                    state = post
                    append_buffer(
                        buffer,
                        {
                            "post": post,
                            "prior": prior,
                            "action": action,
                            "logprob": logprob,
                            "actor_ent": actor_ent,
                            "obs": obs,
                        },
                    )
                batch = build_batch(buffer)
                batch["reward"] = agent.calculate_reward(batch, method="prediction")
                batch["init_state"] = init_state
                agent.saccade_train(batch)

        if i % configs.eval_every == 0:
            openl, mse = agent.saccade_evaluation(batch)
            logger.video("train_openl", to_np(openl))
            logger.scalar("Sac_MSE", mse)

        logger.step = (i + 1) * configs.envs * configs.num_steps

        if i % configs.log_every == 0:
            for name, values in agent._metrics.items():
                logger.scalar(name, float(np.mean(values)))
                agent._metrics[name] = []

        logger.write(fps=True)

        if i % configs.backup_model_every == 0:
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, exp_logdir / "latest.pt")

    print("Training complete.")
    try:
        envs.close()
    except Exception:
        print("Env close failure")
        pass
