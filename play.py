#!/usr/bin/env python3
"""Play a trained DreamerV3 model with live model reloading via pygame window.

Reads the config to determine which environment to create, so a single script
handles all MMJC suites (mmjc, mmjcnavcont, mmjchiernav, mmjcnavego, etc.).

Usage:
    python play.py --logdir logs/run1 --configs mmjc --device cpu
    python play.py --logdir logs/run2 --configs mmjc_nav_sep --device cuda:0
    python play.py --logdir logs/run3 --configs mmjc_hier_nav --device cpu --fps 10
    python play.py --logdir logs/run4 --configs mmjc_nav_ego --device cpu
    python play.py --logdir logs/run5 --configs mmjc_hier_nav_ego --device cpu

Keys:
    0       Switch to random policy
    1-9     Set curriculum stage max_distance (nav envs only)
    Q/ESC   Quit
"""

import os

os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import datetime
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

ACTION_NAMES = ["Forward", "Backward", "CW", "CCW"]


class PlayAgent(torch.nn.Module):
    """Inference-only Dreamer agent with state_dict compatible with training Dreamer."""

    def __init__(self, obs_space, act_space, config):
        super().__init__()
        self._config = config
        self._wm = models.WorldModel(obs_space, act_space, 0, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
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
    key = "_wm.encoder._cnn.layers.0.weight"
    if key in state_dict:
        cnn_depth = state_dict[key].shape[0]
        config.encoder["cnn_depth"] = cnn_depth
        config.decoder["cnn_depth"] = cnn_depth
        print(f"  Inferred cnn_depth={cnn_depth}")

    key = "_wm.dynamics.W"
    if key in state_dict:
        config.dyn_deter = state_dict[key].shape[1]
        print(f"  Inferred dyn_deter={config.dyn_deter}")

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

    actor_linears = sorted(
        k for k in state_dict
        if k.startswith("_task_behavior.actor.layers.Actor_linear")
        and k.endswith(".weight")
    )
    if actor_linears:
        config.actor["layers"] = len(actor_linears)
        print(f"  Inferred actor layers={len(actor_linears)}")

    value_linears = sorted(
        k for k in state_dict
        if k.startswith("_task_behavior.value.layers.Value_linear")
        and k.endswith(".weight")
    )
    if value_linears:
        config.critic["layers"] = len(value_linears)
        print(f"  Inferred critic layers={len(value_linears)}")

    key = "_wm.dynamics._img_in_layers.0.weight"
    if key in state_dict:
        config.dyn_hidden = state_dict[key].shape[0]
        print(f"  Inferred dyn_hidden={config.dyn_hidden}")

    return config


def count_steps(folder):
    """Count environment steps from saved episodes (same as dreamer.py)."""
    folder = pathlib.Path(folder)
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def get_model_mtime(path):
    """Get file modification time, or None if not found."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def format_mtime(mtime):
    """Format mtime as human-readable timestamp."""
    if mtime is None:
        return "?"
    return datetime.datetime.fromtimestamp(mtime).strftime("%H:%M:%S")


def load_weights(agent, path, device, max_retries=10, retry_delay=5.0):
    """Load agent weights with retry logic for concurrent training writes."""
    for attempt in range(max_retries):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state_dict = ckpt["agent_state_dict"]
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


def make_play_env(suite, task, config):
    """Create environment matching dreamer.py's make_env dispatch."""
    if suite == "mmjc":
        raw_env = mmjc.MMJC(task, config.size, render_mode="rgb_array")
        env = wrappers.NormalizeActions(raw_env)
    elif suite == "mmjcnav":
        raw_env = mmjc.MMJCNav(task, config.size, render_mode="rgb_array")
        env = wrappers.NormalizeActions(raw_env)
    elif suite == "mmjcnavcont":
        raw_env = mmjc.MMJCNavCont(task, config.size, render_mode="rgb_array")
        env = wrappers.NormalizeActions(raw_env)
    elif suite == "mmjcnavego":
        raw_env = mmjc.MMJCNavEgo(task, config.size, render_mode="rgb_array")
        env = wrappers.NormalizeActions(raw_env)
    elif suite == "mmjchiernav":
        raw_env = mmjc.MMJCHierNav(
            task, config.size,
            model_path=config.model_path,
            k_steps=config.k_steps,
            hidden_sizes=config.hidden_sizes,
            render_mode="rgb_array",
        )
        env = wrappers.OneHotAction(raw_env)
    elif suite == "mmjchiernavego":
        raw_env = mmjc.MMJCHierNavEgo(
            task, config.size,
            model_path=config.model_path,
            k_steps=config.k_steps,
            hidden_sizes=config.hidden_sizes,
            render_mode="rgb_array",
        )
        env = wrappers.OneHotAction(raw_env)
    else:
        raise NotImplementedError(f"Unknown suite: {suite}")
    return raw_env, env


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
    config.compile = False

    logdir = pathlib.Path(args.logdir).expanduser()
    model_path = logdir / "latest.pt"

    # ── Create environment ─────────────────────────────────────────────
    suite, task = config.task.split("_", 1)
    raw_env, env = make_play_env(suite, task, config)

    # Detect env capabilities
    has_curriculum = hasattr(raw_env, "CURRICULUM_STAGES")
    is_hierarchical = hasattr(raw_env, "substep_callback")
    is_discrete = hasattr(env.action_space, "n")

    config.num_actions = env.action_space.n if is_discrete else env.action_space.shape[0]
    config.time_limit //= config.action_repeat

    print(f"Task: {config.task}  (suite={suite})")
    if is_discrete:
        print(f"Actions: {config.num_actions} ({', '.join(ACTION_NAMES)})")
    else:
        print(f"Actions: {config.num_actions} (continuous)")
    print(f"Model: {model_path}")
    if has_curriculum:
        print(f"Curriculum: {len(raw_env.CURRICULUM_STAGES)} stages")
    if is_hierarchical:
        print(f"Hierarchical: k_steps={config.k_steps}")

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

    # ── Build agent ────────────────────────────────────────────────────
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
    pygame.display.set_caption(f"DreamerV3 Play — {suite}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    print(f"Display: {display_w}x{display_h} (from {frame_w}x{frame_h} frame)")
    print(f"Playing at {args.fps} FPS.  Q=quit", end="")
    if has_curriculum:
        print("  0=random  1-9=curriculum", end="")
    print("\n")

    logdir_name = logdir.name
    traindir = logdir / "train_eps"
    train_steps = count_steps(traindir) if traindir.exists() else 0
    model_time_str = format_mtime(last_mtime)

    episode_count = 0
    reload_count = 0
    use_random = False
    reset_requested = False
    running = True

    # Build stage labels for curriculum envs
    if has_curriculum:
        STAGE_LABELS = ["Random"] + [
            f"S{s['max_distance']:.0f}" if s["max_distance"] != float("inf") else "S-inf"
            for s in raw_env.CURRICULUM_STAGES
        ]
    else:
        STAGE_LABELS = []

    # Mutable HUD state (updated each step; substep callback reads this)
    hud_state = {"step": 0, "reward": 0.0, "last_r": 0.0, "action": ""}

    # ── Rendering helper ───────────────────────────────────────────────
    def render_frame():
        """Render + display one frame. Used as substep callback for hier envs."""
        nonlocal running, use_random, reset_requested, train_steps, model_time_str
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif has_curriculum and event.key == pygame.K_0:
                    use_random = not use_random
                    reset_requested = True
                    print(f"  Policy: {'random' if use_random else 'agent'}")
                elif has_curriculum and pygame.K_1 <= event.key <= pygame.K_9:
                    stage = event.key - pygame.K_1
                    stage = min(stage, len(raw_env.CURRICULUM_STAGES) - 1)
                    raw_env._curriculum_stage = stage
                    max_d = raw_env.CURRICULUM_STAGES[stage]["max_distance"]
                    if max_d == float("inf"):
                        tl = config.time_limit
                    else:
                        tl = min(int(10 * max_d), config.time_limit)
                    raw_env.set_inner_time_limit(tl)
                    use_random = False
                    reset_requested = True
                    print(f"  Curriculum -> stage {stage + 1} (max_dist={max_d}, time_limit={tl})")
        if not running:
            return

        frame = env.render()
        if frame is None:
            return
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (display_w, display_h))
        screen.blit(surf, (0, 0))

        # HUD overlay
        hud_lines = [
            f"{logdir_name}  {train_steps//1000}k steps  @{model_time_str}",
            f"Ep {episode_count + 1}  Step {hud_state['step']}",
            f"Reward: {hud_state['reward']:.2f}  ({hud_state['last_r']:+.3f})",
        ]
        if is_discrete:
            hud_lines.append(f"Action: {hud_state['action']}")
        extra = []
        if has_curriculum:
            if use_random:
                extra.append(STAGE_LABELS[0])
            else:
                extra.append(STAGE_LABELS[raw_env._curriculum_stage + 1])
        extra.append(f"Reloads: {reload_count}")
        if use_random:
            extra.append("RANDOM")
        hud_lines.append("  ".join(extra))

        y = 8
        for line in hud_lines:
            shadow = font.render(line, True, (0, 0, 0))
            text = font.render(line, True, (255, 255, 255))
            screen.blit(shadow, (11, y + 1))
            screen.blit(text, (10, y))
            y += 20

        pygame.display.flip()
        clock.tick(args.fps)

    # Register substep callback for hierarchical envs
    if is_hierarchical:
        raw_env.substep_callback = render_frame

    # ── Main loop ──────────────────────────────────────────────────────
    try:
        while running:
            # ── Reset episode ──────────────────────────────────────────
            obs = env.reset()
            obs = {k: np.array([v]) for k, v in obs.items()}

            state = None
            total_reward = 0.0
            step_count = 0
            done = False

            # ── Episode loop ───────────────────────────────────────────
            while not done and running and not reset_requested:
                # For non-hierarchical envs, handle events + render here
                if not is_hierarchical:
                    render_frame()
                    if not running or reset_requested:
                        break

                # Select action
                if use_random:
                    if is_discrete:
                        action_idx = np.random.randint(config.num_actions)
                        act_np = np.zeros(config.num_actions, dtype=np.float32)
                        act_np[action_idx] = 1.0
                    else:
                        act_np = env.action_space.sample()
                        action_idx = -1
                else:
                    action, state = agent.policy(obs, state)
                    act_np = action[0].cpu().numpy()
                    action_idx = np.argmax(act_np) if is_discrete else -1

                if is_discrete and action_idx < len(ACTION_NAMES):
                    action_name = ACTION_NAMES[action_idx]
                else:
                    action_name = ""

                # Update HUD state before stepping (substep callback reads this)
                hud_state["step"] = step_count + 1
                hud_state["reward"] = total_reward
                hud_state["action"] = action_name

                obs, reward, done, info = env.step(act_np)
                total_reward += reward
                hud_state["reward"] = total_reward
                hud_state["last_r"] = reward
                step_count += 1

                obs = {k: np.array([v]) for k, v in obs.items()}

            # ── End of episode ─────────────────────────────────────────
            if not running:
                break

            if reset_requested:
                reset_requested = False
                state = None
                continue

            episode_count += 1
            ep_msg = f"Ep {episode_count}: reward={total_reward:.2f}  steps={step_count}"
            if "curriculum_stage" in info:
                ep_msg += f"  curriculum={info['curriculum_stage']}"
            print(ep_msg)

            # Check for model update
            mtime = get_model_mtime(model_path)
            if mtime is not None and mtime != last_mtime:
                print("  Model updated, reloading...")
                if load_weights(agent, model_path, config.device):
                    last_mtime = mtime
                    reload_count += 1
                    train_steps = count_steps(traindir) if traindir.exists() else 0
                    model_time_str = format_mtime(mtime)
                    print(f"  Reloaded! (reload #{reload_count}, {train_steps//1000}k steps)")
                else:
                    print("  Reload failed, using previous weights")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit()
        env.close()


if __name__ == "__main__":
    main()
