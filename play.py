#!/usr/bin/env python3
"""Play a trained DreamerV3 model with live model reloading via pygame window.

Usage:
    python play.py --logdir logs/maz-017-rev0 --configs mmjc --device cpu
    python play.py --logdir logs/maz-017-rev0 --configs mmjc --device cuda:1 --fps 30

Supports live model reloading - checks for checkpoint updates after each episode
and safely reloads the latest weights, even while training is running.
"""

import os

os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np
import pygame
import ruamel.yaml as yaml
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import envs.mmjc as mmjc
import envs.wrappers as wrappers
import exploration as expl
import models
import tools


class PlayAgent(torch.nn.Module):
    """Inference-only Dreamer agent with state_dict compatible with training Dreamer."""

    def __init__(self, obs_space, act_space, config):
        super().__init__()
        self._config = config
        self._wm = models.WorldModel(obs_space, act_space, 0, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        # Must match Dreamer's _expl_behavior structure for state_dict loading
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(config.device)

    @torch.no_grad()
    def policy(self, obs, state):
        """Run deterministic policy inference."""
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
        actor = self._task_behavior.actor(feat)
        action = actor.mode()
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        state = (latent, action)
        return action, state


def infer_config_from_checkpoint(state_dict, config):
    """Override config model-size settings to match checkpoint architecture."""
    # CNN depth: first conv layer shape is [cnn_depth, in_channels, k, k]
    key = "_wm.encoder._cnn.layers.0.weight"
    if key in state_dict:
        cnn_depth = state_dict[key].shape[0]
        config.encoder["cnn_depth"] = cnn_depth
        config.decoder["cnn_depth"] = cnn_depth
        print(f"  Inferred cnn_depth={cnn_depth}")

    # Deterministic state size: dynamics.W shape is [1, deter]
    key = "_wm.dynamics.W"
    if key in state_dict:
        config.dyn_deter = state_dict[key].shape[1]
        print(f"  Inferred dyn_deter={config.dyn_deter}")

    # Hidden units and layers from reward head
    reward_linears = sorted(
        k for k in state_dict
        if k.startswith("_wm.heads.reward.layers.Reward_linear")
        and k.endswith(".weight")
    )
    if reward_linears:
        config.units = state_dict[reward_linears[0]].shape[0]
        n_layers = len(reward_linears)
        config.reward_head["layers"] = n_layers
        config.cont_head["layers"] = n_layers
        print(f"  Inferred units={config.units}, reward/cont layers={n_layers}")

    # Actor layers
    actor_linears = sorted(
        k for k in state_dict
        if k.startswith("_task_behavior.actor.layers.Actor_linear")
        and k.endswith(".weight")
    )
    if actor_linears:
        config.actor["layers"] = len(actor_linears)
        print(f"  Inferred actor layers={len(actor_linears)}")

    # Value/critic layers
    value_linears = sorted(
        k for k in state_dict
        if k.startswith("_task_behavior.value.layers.Value_linear")
        and k.endswith(".weight")
    )
    if value_linears:
        config.critic["layers"] = len(value_linears)
        print(f"  Inferred critic layers={len(value_linears)}")

    # dyn_hidden from GRU input layer
    key = "_wm.dynamics._img_in_layers.0.weight"
    if key in state_dict:
        config.dyn_hidden = state_dict[key].shape[0]
        print(f"  Inferred dyn_hidden={config.dyn_hidden}")

    return config


def get_model_mtime(path):
    """Get file modification time, or None if not found."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def load_weights(agent, path, device, max_retries=10, retry_delay=5.0):
    """Load agent weights with retry logic for concurrent training writes."""
    for attempt in range(max_retries):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state_dict = ckpt["agent_state_dict"]
            # torch.compile adds '_orig_mod.' prefix to keys — strip it
            state_dict = {
                k.replace("._orig_mod", ""): v for k, v in state_dict.items()
            }
            agent.load_state_dict(state_dict)
            return True
        except (RuntimeError, EOFError, KeyError, pickle.UnpicklingError) as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed ({type(e).__name__}), retrying...")
                time.sleep(retry_delay)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return False
        except FileNotFoundError:
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Play DreamerV3 with live reload")
    parser.add_argument("--logdir", required=True, help="Experiment log directory")
    parser.add_argument("--configs", nargs="+", help="Config sections from configs.yaml")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda:0/etc)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for rendering")
    parser.add_argument(
        "--display_size", type=int, default=512, help="Display height in pixels"
    )
    args, remaining = parser.parse_known_args()

    # ── Load config (same mechanism as dreamer.py) ──────────────────────
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *(args.configs or [])]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    config_parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        config_parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = config_parser.parse_args(remaining)
    config.device = args.device
    config.compile = False  # No torch.compile for play mode

    logdir = pathlib.Path(args.logdir).expanduser()
    model_path = logdir / "latest.pt"

    # ── Create environment (same wrapper chain as dreamer.py) ──────────
    suite, task = config.task.split("_", 1)
    env = mmjc.MMJC(task, config.size, render_mode="rgb_array")
    env = wrappers.NormalizeActions(env)

    config.num_actions = env.action_space.shape[0]
    config.time_limit //= config.action_repeat

    print(f"Task: {config.task}")
    print(f"Actions: {config.num_actions}")
    print(f"Model: {model_path}")

    # ── Infer model architecture from checkpoint ───────────────────────
    print("Waiting for checkpoint...")
    while not model_path.exists():
        print(f"  {model_path} not found, waiting...")
        time.sleep(5.0)

    print("Inferring model architecture from checkpoint...")
    ckpt = torch.load(model_path, map_location=config.device, weights_only=False)
    state_dict = {
        k.replace("._orig_mod", ""): v
        for k, v in ckpt["agent_state_dict"].items()
    }
    config = infer_config_from_checkpoint(state_dict, config)

    # ── Build agent (architecture now matches checkpoint) ──────────────
    agent = PlayAgent(env.observation_space, env.action_space, config).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    agent.load_state_dict(state_dict)
    last_mtime = get_model_mtime(model_path)
    print("Checkpoint loaded!")

    # ── Compute display dimensions ─────────────────────────────────────
    probe_obs = env.reset()
    probe_frame = env.render()
    frame_h, frame_w = probe_frame.shape[:2]
    aspect = frame_w / frame_h
    display_h = args.display_size
    display_w = int(display_h * aspect)

    # ── Initialize pygame ──────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("DreamerV3 Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    print(f"Playing at {args.fps} FPS. Press Q or close window to stop.\n")

    episode_count = 0
    reload_count = 0
    running = True

    try:
        while running:
            # ── Reset episode ───────────────────────────────────────────
            obs = env.reset()
            obs = {k: np.array([v]) for k, v in obs.items()}

            state = None
            total_reward = 0.0
            step_count = 0
            done = False

            # ── Episode loop ────────────────────────────────────────────
            while not done and running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                if not running:
                    break

                # Agent action
                action, state = agent.policy(obs, state)
                act_np = action[0].cpu().numpy()

                # Step environment (NormalizeActions handles denormalization)
                obs, reward, done, info = env.step(act_np)
                total_reward += reward
                step_count += 1

                # Add batch dimension for agent
                obs = {k: np.array([v]) for k, v in obs.items()}

                # ── Render frame ────────────────────────────────────────
                frame = env.render()
                # numpy (H, W, 3) -> pygame surface expects (W, H, 3)
                surf = pygame.surfarray.make_surface(
                    np.transpose(frame, (1, 0, 2))
                )
                surf = pygame.transform.scale(surf, (display_w, display_h))
                screen.blit(surf, (0, 0))

                # HUD overlay
                hud_lines = [
                    f"Ep {episode_count + 1}  Step {step_count}",
                    f"Reward: {total_reward:.2f}  ({reward:+.3f})",
                    f"Reloads: {reload_count}",
                ]
                y = 8
                for line in hud_lines:
                    shadow = font.render(line, True, (0, 0, 0))
                    text = font.render(line, True, (255, 255, 255))
                    screen.blit(shadow, (11, y + 1))
                    screen.blit(text, (10, y))
                    y += 20

                pygame.display.flip()
                clock.tick(args.fps)

            # ── End of episode ──────────────────────────────────────────
            if not running:
                break

            episode_count += 1
            print(
                f"Ep {episode_count}: reward={total_reward:.2f}  steps={step_count}"
            )

            # Check for model update
            mtime = get_model_mtime(model_path)
            if mtime is not None and mtime != last_mtime:
                print("  Model updated, reloading...")
                if load_weights(agent, model_path, config.device):
                    last_mtime = mtime
                    reload_count += 1
                    print(f"  Reloaded! (reload #{reload_count})")
                else:
                    print("  Reload failed, using previous weights")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit()
        env.close()


if __name__ == "__main__":
    main()
