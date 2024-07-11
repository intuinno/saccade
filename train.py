import numpy as np
import argparse
from ruamel.yaml import YAML
import torch
import pathlib
import sys
from tqdm import tqdm

import tools
from l2hwm import L2HWM
from datetime import datetime
import pytz
from envs.vec_sac_env import VecSaccadeEnvAdapter


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
    config.time_limit //= config.action_repeat
    config.use_amp = True if config.precision == 16 else False

    print("Logdir:", logdir)
    logdir.mkdir(parents=True, exist_ok=False)

    print("Create envs.")
    vec_envs = VecSaccadeEnvAdapter(config.batch_size, config)
    acts = vec_envs.action_space
    print("Action space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    config_backup = argparse.Namespace(**vars(config))
    config_backup.traindir = str(config.traindir)
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(logdir / "config.yml", "w") as f:
        yaml.dump(vars(config_backup), f)

    # Build model
    model = L2HWM(config).to(config.device)

    print(f"========== Using {config.device} device ===================")

    # Load model if args.load_model is not none
    if args.load_model is not None:
        model_path = pathlib.Path(args.load_model).expanduser()
        print(f"========== Loading saved model from {model_path} ===========")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Build logger
    logger = tools.Logger(exp_logdir, 0)
    metrics = {}

    for epoch in range(configs.num_epochs):
        # Write evaluation summary
        print(f"======== Epoch {epoch} / {configs.num_epochs} ==========")
        now = datetime.now(tz)
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        logger.step = epoch
        if epoch % configs.eval_every == 0:
            print(f"Evaluating ... ")
            recon_loss_list = []
            for i, x in enumerate(tqdm(val_dataloader)):
                openl, recon_loss = model.video_pred(x.to(configs.device))
                if i == 0:
                    logger.video("eval_openl", openl)
                recon_loss_list.append(recon_loss)
            recon_loss_mean = np.mean(recon_loss_list)
            logger.scalar("eval_video_nll", recon_loss_mean)

        print(f"Training ...")
        for i, x in enumerate(tqdm(train_dataloader)):
            x = x.to(configs.device)
            met = model.local_train(x)
            for name, values in met.items():
                if not name in metrics.keys():
                    metrics[name] = [values]
                else:
                    metrics[name].append(values)

        # Write training summary
        for name, values in metrics.items():
            logger.scalar(name, float(np.mean(values)))
            metrics[name] = []
        if epoch % configs.train_gif_every == 0:
            openl, recon_loss = model.video_pred(x)
            logger.video("train_openl", openl)
            logger.write(fps=True)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            # 'logger': logger,
        }
        # Save Check point
        if epoch % configs.save_model_every == 0:
            torch.save(checkpoint, exp_logdir / "latest_checkpoint.pt")

        if epoch % configs.backup_model_every == 0:
            torch.save(checkpoint, exp_logdir / f"state_{epoch}.pt")

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
