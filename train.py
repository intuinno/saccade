import numpy as np
import argparse
from ruamel.yaml import YAML
import torch
import pathlib
import sys
from tqdm import tqdm

import tools
from datetime import datetime
import pytz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--load_model", type=str)
    args, remaining = parser.parse_known_args()
    rootdir = pathlib.Path(sys.argv[0]).parent
    yaml = YAML(typ='safe')
    configs = yaml.load(
        rootdir / "configs.yml"
    ).read_text()

    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    configs = parser.parse_args(remaining)

    exp_name = configs.exp_name + "_"
    tools.set_seed_everywhere(configs.seed)

    tz = pytz.timezone("US/Pacific")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    exp_name += date_time

    # Creating model dir with experiment name.
    exp_logdir = rootdir / configs.logdir / configs.dataset / exp_name
    print("Logdir", exp_logdir)
    exp_logdir.mkdir(parents=True, exist_ok=True)

    # Dumping config.
    with open(exp_logdir / "config.yml", "w") as f:
        yaml.dump(vars(configs), f, default_flow_style=False)


    # Load dataset.
    train_dataset, val_dataset = load_dataset(configs)
    train_dataloader = DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # Build model
    model = L2HWM(configs).to(configs.device)

    print(f"========== Using {configs.device} device ===================")

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
