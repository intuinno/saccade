#!/usr/bin/env python3

import argparse
import pathlib
import sys
import time
import torch
import numpy as np
import ruamel.yaml as yaml

# Configure OpenGL for rendering
import os

os.environ["MUJOCO_GL"] = "glfw"


sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent / "dreamerv3"))

import models
import tools
import envs.wrappers as wrappers


class Config:
    """Simple configuration class that acts like argparse.Namespace"""

    def __init__(self, config_dict=None):
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)

    def update(self, update_dict):
        """Update configuration with new values"""
        new_config = Config()
        # Copy existing attributes
        for key, value in self.__dict__.items():
            setattr(new_config, key, value)
        # Update with new values (recursively for dicts)
        for key, value in update_dict.items():
            if (
                hasattr(new_config, key)
                and isinstance(getattr(new_config, key), dict)
                and isinstance(value, dict)
            ):
                # Recursive update for nested dicts
                existing_value = getattr(new_config, key)
                updated_value = existing_value.copy()
                updated_value.update(value)
                setattr(new_config, key, updated_value)
            else:
                setattr(new_config, key, value)
        return new_config


def load_config(configs):
    """Load configuration from configs.yaml"""
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs_yaml = yaml_loader.load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    # Start with defaults
    config = Config(configs_yaml["defaults"])

    # Override with specific configs
    for name in configs:
        if name in configs_yaml:
            config = config.update(configs_yaml[name])

    return config


def make_env(config, mode, id, render_mode=None, temp_k=4):
    """Create environment directly (simplified version from dreamer.py)"""
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
            grayscale=config.grayscale,
            life_done=True,
            sticky=config.stickey,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "vertebrate":
        import envs.vertebrate_env as vertebrate_env

        # For vertebrate env, we need to modify the constructor to support render_mode
        # Use configurable temp_k for more responsive rendering
        if render_mode == "human":
            env = vertebrate_env.VertebrateEnv(
                seed=config.seed + id,
                render_mode="human",
                temp_k=temp_k,
                model_name=config.model_path,
            )
        else:
            env = vertebrate_env.VertebrateEnv(seed=config.seed + id, temp_k=temp_k)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)

    env = wrappers.TimeLimit(env, config.time_limit)
    return env


class SimpleAgent:
    """Simplified agent class for inference only"""

    def __init__(self, obs_space, act_space, config):
        self._step = 0

        # Force CPU device for model creation - patch config temporarily
        original_device = getattr(config, "device", None)
        config.device = "cpu"

        # Create a config copy that always returns 'cpu' for any device-related attribute
        class CPUOnlyConfig:
            def __init__(self, original_config):
                self._original_config = original_config
                # Copy all attributes from original config
                for key, value in vars(original_config).items():
                    setattr(self, key, value)
                # Override device
                self.device = "cpu"

            def __getattr__(self, name):
                if "device" in name.lower():
                    return "cpu"
                return getattr(self._original_config, name)

        cpu_config = CPUOnlyConfig(config)

        self._wm = models.WorldModel(obs_space, act_space, self._step, cpu_config)
        self._task_behavior = models.ImagBehavior(cpu_config, self._wm)

        # Restore original device if it existed
        if original_device is not None:
            config.device = original_device

        self.training = False
        self._state = None
        self._prev_action = None

    def _preprocess_obs(self, key, value):
        """Preprocess observations to ensure correct data types"""
        tensor = torch.tensor(value)

        # Convert uint8 images to float32 and normalize to [0, 1]
        if key == "image" and tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor

    def _preprocess_tensor_obs(self, key, tensor):
        """Preprocess tensor observations to ensure correct data types"""
        # Convert uint8 images to float32 and normalize to [0, 1]
        if key == "image" and tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor

    def __call__(self, obs, training=False):
        obs = {
            k: (
                self._preprocess_obs(k, v).unsqueeze(0)  # Add batch dimension only
                if not torch.is_tensor(v)
                else self._preprocess_tensor_obs(k, v).unsqueeze(0)
            )
            for k, v in obs.items()
            if "log_" not in k
        }

        with torch.no_grad():
            embed = self._wm.encoder(obs)

            # Handle state and action for single-step inference
            is_first = self._state is None
            batch_size = embed.shape[0]

            if is_first:
                # Initialize previous action for first step
                self._prev_action = torch.zeros(
                    batch_size, 3, device=embed.device
                )  # 3 actions
                is_first_tensor = torch.ones(
                    batch_size, dtype=torch.bool, device=embed.device
                )
            else:
                is_first_tensor = torch.zeros(
                    batch_size, dtype=torch.bool, device=embed.device
                )

            # Use obs_step for single-step inference
            self._state, _ = self._wm.dynamics.obs_step(
                self._state, self._prev_action, embed.squeeze(1), is_first_tensor
            )

            feat = self._wm.dynamics.get_feat(self._state)
            action = self._task_behavior.actor(feat).sample()

            # Store action for next step
            if isinstance(action, dict) and "action" in action:
                self._prev_action = action["action"]
            else:
                self._prev_action = action

            # Convert to numpy and remove batch dimension
            if isinstance(action, dict):
                action_out = {k: v.squeeze(0).cpu().numpy() for k, v in action.items()}
            else:
                action_out = action.squeeze(0).cpu().numpy()

        return action_out

    def state_dict(self):
        return {
            "wm": self._wm.state_dict(),
            "task_behavior": self._task_behavior.state_dict(),
        }

    def load_state_dict(self, state_dict):
        def strip_compiled_prefix_nested(state_dict):
            """Remove _orig_mod. prefix from compiled model state dict, handling nested structure"""
            new_state_dict = {}
            for key, value in state_dict.items():
                # Handle nested _orig_mod. patterns like "_wm._orig_mod.encoder..."
                if "._orig_mod." in key:
                    # Find and remove the ._orig_mod. part
                    parts = key.split("._orig_mod.")
                    if len(parts) == 2:
                        new_key = parts[0] + "." + parts[1]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict

        if "agent_state_dict" in state_dict:
            # Handle full checkpoint format
            full_state = state_dict["agent_state_dict"]

            # Strip compiled model prefix if present
            full_state = strip_compiled_prefix_nested(full_state)

            wm_state = {k[4:]: v for k, v in full_state.items() if k.startswith("_wm.")}
            behavior_state = {
                k[15:]: v
                for k, v in full_state.items()
                if k.startswith("_task_behavior.")
            }
            self._wm.load_state_dict(wm_state)
            self._task_behavior.load_state_dict(behavior_state)
        else:
            # Handle simple format
            wm_state = strip_compiled_prefix_nested(state_dict.get("wm", {}))
            behavior_state = strip_compiled_prefix_nested(
                state_dict.get("task_behavior", {})
            )
            self._wm.load_state_dict(wm_state)
            self._task_behavior.load_state_dict(behavior_state)

    def eval(self):
        self.training = False
        self._wm.eval()
        self._task_behavior.eval()


def create_render_env(config, env_id=0, temp_k=4):
    """Create a single environment with rendering enabled"""
    return make_env(
        config,
        "eval",
        env_id,
        render_mode="human",
        temp_k=temp_k,
    )


def play_model(config, temp_k=4):
    """Load model and play interactively with rendering"""

    # Setup checkpoint path - can be direct model path or logdir
    if hasattr(config, "model_path") and config.model_path:
        checkpoint_path = pathlib.Path(config.model_path).expanduser()
        print(f"Using direct model path: {checkpoint_path}")
    else:
        # Fallback to logdir/latest.pt
        logdir = (
            pathlib.Path(config.logdir).expanduser()
            if config.logdir
            else pathlib.Path.cwd() / "logdir"
        )
        checkpoint_path = logdir / "latest.pt"
        print(f"Looking for checkpoint in logdir: {logdir}")

    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        if hasattr(config, "model_path") and config.model_path:
            print("Make sure the model file path is correct.")
        else:
            print(
                "Make sure training has created a checkpoint file or specify --model_path."
            )
        return

    print(f"✅ Found checkpoint: {checkpoint_path}")

    # Create environment with rendering
    print("Creating environment with rendering...")
    try:
        env = create_render_env(config, temp_k=temp_k)
        print(f"✅ Environment created: {type(env).__name__}")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space keys: {list(env.observation_space.spaces.keys())}")

        # Set num_actions from action space (similar to dreamer.py)
        acts = env.action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        print(f"   Number of actions: {config.num_actions}")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create agent
    print("Creating agent...")
    try:
        # Use standard gym spaces directly
        obs_space = env.observation_space
        act_space = env.action_space

        # Ensure device is set to CPU for all components before creating agent
        # This is critical because MLP class in networks.py uses device parameter directly
        config.device = "cpu"

        # Force CPU for all possible device-related config attributes
        device_attrs = ["device", "model_device", "network_device", "torch_device"]
        for attr in device_attrs:
            if hasattr(config, attr):
                setattr(config, attr, "cpu")

        # Also check for nested configs that might have device settings
        for attr_name in dir(config):
            if not attr_name.startswith("_"):
                attr_value = getattr(config, attr_name)
                if isinstance(attr_value, dict):
                    for key in attr_value:
                        if "device" in key.lower():
                            attr_value[key] = "cpu"

        print(f"   Final config device: {config.device}")

        # Monkey-patch networks to force CPU usage
        import networks

        original_MLP_init = networks.MLP.__init__

        def cpu_MLP_init(self, *args, **kwargs):
            # Force device to CPU in kwargs
            kwargs["device"] = "cpu"
            # Also check if device is passed positionally (12th parameter, index 11)
            if len(args) > 11:
                args = list(args)
                args[11] = "cpu"
                args = tuple(args)
            print(f"   MLP init patched: device set to CPU")
            return original_MLP_init(self, *args, **kwargs)

        # Apply the monkey-patch
        networks.MLP.__init__ = cpu_MLP_init

        try:
            agent = SimpleAgent(obs_space, act_space, config)
        finally:
            # Restore original MLP init
            networks.MLP.__init__ = original_MLP_init
        print(f"✅ Agent created")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback

        traceback.print_exc()
        return

    # Load checkpoint
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        # Handle different checkpoint formats
        if "agent_state_dict" in checkpoint:
            print("Found full checkpoint format with agent_state_dict")
            agent_state = checkpoint["agent_state_dict"]
        else:
            print("Using direct checkpoint format")
            agent_state = checkpoint

        agent.load_state_dict({"agent_state_dict": agent_state})
        agent.eval()  # Set to evaluation mode
        print(f"✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return

    # Play episodes
    print("\n🎮 Starting interactive play...")
    print("Press Ctrl+C to stop")

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- Episode {episode} ---")

            # Reset environment
            obs = env.reset()
            if callable(obs):
                obs = obs()

            # Render initial state
            env.render()

            total_reward = 0
            step = 0
            done = False

            while not done:
                step += 1

                # Get action from agent
                with torch.no_grad():
                    action = agent(obs, training=False)

                # Extract action if it's a dictionary (for discrete action spaces)
                if isinstance(action, dict) and "action" in action:
                    action = action["action"]
                elif isinstance(action, dict):
                    # For other action formats, take the first value
                    action = next(iter(action.values()))

                # Step environment
                obs, reward, done, info = env.step(action)
                if callable(obs):
                    obs = obs()

                # Render the environment
                env.render()

                total_reward += reward

                # Print progress
                if step % 10 == 0 or done:
                    print(
                        f"  Step {step:3d}: Reward {reward:6.3f}, Total: {total_reward:8.3f}"
                    )

                # Add small delay to make it watchable
                time.sleep(0.05)

                # Safety check
                if step > 1000:
                    print("  Episode too long, terminating...")
                    break

            print(
                f"Episode {episode} finished: {step} steps, {total_reward:.3f} total reward"
            )

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    finally:
        # Cleanup
        try:
            env.close()
        except:
            pass

    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(description="Play trained model with rendering")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["defaults"],
        help="Configuration names to use (e.g., vertebrate)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory containing the checkpoint (latest.pt) - used if --model_path not specified",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Direct path to model file (e.g., models/model.pt) - takes priority over --logdir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--temp_k",
        type=int,
        default=16,
        help="Temporal abstraction parameter (lower = more responsive, default: 16)",
    )

    args = parser.parse_args()

    print(f"🎯 Loading configuration: {args.configs}")

    # Load configuration
    try:
        config = load_config(args.configs)

        # Override with command line args
        if args.logdir:
            config.logdir = args.logdir
        if args.model_path:
            config.model_path = args.model_path

        # Force device setting to command line argument everywhere
        original_device = getattr(config, "device", "unknown")
        setattr(config, "device", args.device)
        print(f"   Changed main device from {original_device} to {args.device}")

        # Recursively update all device settings in nested structures
        def update_device_recursive(obj, device_value, path=""):
            updated_count = 0
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == "device" and value != device_value:
                        print(f"   Updated {current_path}: {value} -> {device_value}")
                        obj[key] = device_value
                        updated_count += 1
                    else:
                        updated_count += update_device_recursive(
                            value, device_value, current_path
                        )
            elif hasattr(obj, "__dict__"):
                for attr_name in vars(obj):
                    if not attr_name.startswith("_"):
                        current_path = f"{path}.{attr_name}" if path else attr_name
                        attr_value = getattr(obj, attr_name)
                        updated_count += update_device_recursive(
                            attr_value, device_value, current_path
                        )
            return updated_count

        # Apply recursive device update
        updates = update_device_recursive(config, args.device)
        print(f"   Total device settings updated: {updates}")

        # Double-check: ensure no CUDA devices remain
        def find_cuda_devices(obj, path=""):
            cuda_refs = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == "device" and str(value).startswith("cuda"):
                        cuda_refs.append(f"{current_path}: {value}")
                    else:
                        cuda_refs.extend(find_cuda_devices(value, current_path))
            elif hasattr(obj, "__dict__"):
                for attr_name in vars(obj):
                    if not attr_name.startswith("_"):
                        current_path = f"{path}.{attr_name}" if path else attr_name
                        attr_value = getattr(obj, attr_name)
                        cuda_refs.extend(find_cuda_devices(attr_value, current_path))
            elif isinstance(obj, str) and obj.startswith("cuda"):
                cuda_refs.append(f"{path}: {obj}")
            return cuda_refs

        remaining_cuda = find_cuda_devices(config)
        if remaining_cuda:
            print(f"   ⚠️ Warning: Still found CUDA references: {remaining_cuda}")
        else:
            print(f"   ✅ All device settings confirmed as CPU")

        print(f"✅ Configuration loaded")
        print(f"   Task: {config.task}")
        print(f"   Device: {config.device}")
        print(f"   Temporal abstraction (temp_k): {args.temp_k}")
        if hasattr(config, "model_path") and config.model_path:
            print(f"   Model path: {config.model_path}")
        else:
            print(f"   Logdir: {config.logdir}")

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        import traceback

        traceback.print_exc()
        return

    # Play the model
    play_model(config, temp_k=args.temp_k)


if __name__ == "__main__":
    main()
