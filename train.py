import numpy as np
import argparse
from ruamel.yaml import YAML
import torch
import pathlib
import sys
from tqdm import tqdm
from torch.nn import functional as F

import tools
from cognitive_architecture import CognitiveArchitecture
from datetime import datetime
import pytz
from envs.vec_sac_env import VecSaccadeEnv


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


def build_batch_behavior(buffer):
    for key, val in buffer.items():
        if isinstance(val, dict):
            buffer[key] = build_batch_behavior(buffer[key])
        else:
            buffer[key] = torch.stack(buffer[key])
    return buffer


def get_location(action, obs):
    current_x = torch.argmax(obs["loc"], dim=1) % 4
    current_y = torch.argmax(obs["loc"], dim=1) // 4
    delta_x = torch.argmax(action["delta_x"], dim=1) - 3
    delta_y = torch.argmax(action["delta_y"], dim=1) - 3
    loc_x = torch.clip(current_x + delta_x, 0, 3)
    loc_y = torch.clip(current_y + delta_y, 0, 3)
    loc = loc_x + loc_y * 4
    flat_action = F.one_hot(loc, num_classes=16)
    return flat_action


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
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.use_amp = True if config.precision == 16 else False

    print("Logdir:", logdir)
    logdir.mkdir(parents=True, exist_ok=False)

    print("Create envs.")
    vec_envs = VecSaccadeEnv(batch_size=config.batch_size, device=config.device)
    print("Action space", config.action_space)
    config.num_actions = sum(k for k in config.action_space.values())

    config_backup = argparse.Namespace(**vars(config))
    config_backup.logdir = str(config.logdir)
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(logdir / "config.yml", "w") as f:
        yaml.dump(vars(config_backup), f)

    # Build model
    model = CognitiveArchitecture(config).to(config.device)

    print(f"========== Using {config.device} device ===================")

    # Load model if args.load_model is not none
    if config.load_model != "":
        model_path = pathlib.Path(config.load_model).expanduser()
        print(f"========== Loading saved model from {model_path} ===========")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Build logger
    logger = tools.Logger(logdir, 0)
    metrics = {}

    for epoch in tqdm(range(config.num_epochs)):
        state = model.wm.init()
        feat = model.wm.get_feat(state)
        with torch.cuda.amp.autocast(config.use_amp):
            with tools.RequiresGrad(model):
                buffer = {}
                buffer["feat"] = [feat]
                # buffer["action"] = []
                buffer["logprob"] = []
                buffer["actor_ent"] = []
                obs_buffer = []
                buffer["reward"] = []
                obs = vec_envs.reset()
                for _ in range(config.batch_length):
                    action, logprob, actor_ent = model.get_action(feat)
                    detached_action = {k: v.detach().clone() for k, v in action.items()}
                    flat_action = get_location(detached_action, obs)
                    obs, reward, _ = vec_envs.step(flat_action)

                    feat = model.wm_step(flat_action, obs)
                    buffer["feat"].append(feat)
                    # buffer["action"].append(flat_action)
                    buffer["logprob"].append(logprob)
                    buffer["actor_ent"].append(actor_ent)
                    obs_buffer.append(obs)
                    buffer["reward"].append(reward)

                # met = model.train(batch)
                batch = build_batch_behavior(buffer)
                met, recon = model.train(batch)
                for name, values in met.items():
                    if not name in metrics.keys():
                        metrics[name] = [values]
                    else:
                        metrics[name].append(values)
                video_loss, train_video = model.train_video(buffer, obs_buffer)
                logger.video("top_video", train_video)
                logger.scalar("video_loss", float(video_loss))
                logger.scalar("train_batch_mean_reward", batch["reward"].mean().cpu())

        # Write training summary
        for name, values in metrics.items():
            logger.scalar(name, float(np.mean(values)))
            metrics[name] = []
            logger.write(fps=True, step=epoch)
        if epoch % config.train_gif_every == 0:
            openl = model.decode_video(recon, obs_buffer)
            for name, vid in openl.items():
                logger.video(f"{name}-video", vid)
            # Build a video from imaginary central module
            state = model.wm.init()
            feat = model.wm.get_feat(state)
            buffer = {}
            buffer["obs"] = []
            buffer["central_obs"] = []
            obs = vec_envs.reset()
            for _ in range(config.eval_obs_length):
                action, logprob, actor_ent = model.get_action(feat)
                detached_action = {k: v.detach().clone() for k, v in action.items()}
                flat_action = get_location(detached_action, obs)
                central_obs = model.wm.scan_central()

                obs, _, _ = vec_envs.step(flat_action)
                feat = model.wm_step(flat_action, obs)
                buffer["central_obs"].append(central_obs)
                buffer["obs"].append(obs)

            with tools.ImagineMode(model.wm):
                for _ in range(config.eval_img_length):
                    action, _, _ = model.get_action(feat)
                    detached_action = {k: v.detach().clone() for k, v in action.items()}
                    flat_action = get_location(detached_action, obs)
                    central_obs = model.wm.scan_central()

                    obs, _, _ = vec_envs.step(flat_action)
                    feat = model.wm_step(flat_action, obs)
                    buffer["central_obs"].append(central_obs)
                    buffer["obs"].append(obs)

            central_video = tools.central_video(buffer["central_obs"], buffer["obs"])
            logger.video("central-scan", central_video)

        logger.write(fps=True, step=epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            # 'logger': logger,
        }
        # Save Check point
        if epoch % config.save_model_every == 0:
            torch.save(checkpoint, logdir / "latest_checkpoint.pt")

        if epoch % config.backup_model_every == 0:
            torch.save(checkpoint, logdir / f"state_{epoch}.pt")

    print("Training complete.")


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
