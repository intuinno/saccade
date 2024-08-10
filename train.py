import numpy as np
import argparse
from ruamel.yaml import YAML
import torch
import pathlib
import sys
from tqdm import tqdm

import tools
from cognitive_architecture import CognitiveArchitecture
from datetime import datetime
import pytz
from envs.vec_sac_env import VecSaccadeEnvAdapter


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
    vec_envs = VecSaccadeEnvAdapter(config.batch_size, config)
    acts = vec_envs.action_space
    print("Action space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

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
                buffer["action"] = []
                buffer["logprob"] = []
                buffer["actor_ent"] = []
                buffer["obs"] = []
                obs = vec_envs.reset()
                for _ in range(config.batch_length):
                    action, logprob, actor_ent = model.get_action(feat)
                    detached_action = action.detach().clone()
                    obs, _, _, _ = vec_envs.step(detached_action)
                    feat = model.wm_step(detached_action, obs)
                    buffer["feat"].append(feat)
                    buffer["action"].append(action)
                    buffer["logprob"].append(logprob)
                    buffer["actor_ent"].append(actor_ent)
                    buffer["obs"].append(obs)

                # batch["reward"] = model.calculate_reward(batch)
                # met = model.train(batch)
                met, recon = model.train(buffer)
                for name, values in met.items():
                    if not name in metrics.keys():
                        metrics[name] = [values]
                    else:
                        metrics[name].append(values)
                video_loss, train_video = model.train_video(buffer)
                logger.video("top_video", train_video)
                logger.scalar("video_loss", float(video_loss))

        # Write training summary
        for name, values in metrics.items():
            logger.scalar(name, float(np.mean(values)))
            metrics[name] = []
            logger.write(step=epoch)
        if epoch % config.train_gif_every == 0:
            openl = model.decode_video(recon, buffer)
            for name, vid in openl.items():
                logger.video(f"{name}-video", vid)
        logger.write(fps=True)

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
