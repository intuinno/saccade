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
os.environ["MUJOCO_GL"] = "osmesa"

# Disable OpenGL error checking for better compatibility
try:
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
except ImportError:
    pass

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent / "dreamerv3"))

import models
import tools
import envs.wrappers as wrappers


def load_config(configs):
    """Load configuration from configs.yaml"""
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    configs_yaml = yaml_loader.load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )
    
    # Start with defaults
    config = tools.Config(configs_yaml["defaults"])
    
    # Override with specific configs
    for name in configs:
        if name in configs_yaml:
            config = config.update(configs_yaml[name])
    
    return config


def make_env(config, mode, id):
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
        env = vertebrate_env.VertebrateEnv(seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    
    env = wrappers.TimeLimit(env, config.time_limit)
    return env


class SimpleAgent:
    """Simplified agent class for inference only"""
    def __init__(self, obs_space, act_space, config):
        self._step = 0
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        self.training = False
    
    def __call__(self, obs, training=False):
        obs = {k: torch.tensor(v).unsqueeze(0) if not torch.is_tensor(v) else v.unsqueeze(0) 
               for k, v in obs.items() if 'log_' not in k}
        
        with torch.no_grad():
            embed = self._wm.encoder(obs)
            latent, _ = self._wm.rssm.observe(embed, None, None)
            feat = self._wm.rssm.get_feat(latent)
            action = self._task_behavior.actor(feat).sample()
            
            # Convert to numpy and remove batch dimension
            action = {k: v.squeeze(0).cpu().numpy() for k, v in action.items()}
            
        return action
    
    def state_dict(self):
        return {
            'wm': self._wm.state_dict(),
            'task_behavior': self._task_behavior.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        if 'agent_state_dict' in state_dict:
            # Handle full checkpoint format
            full_state = state_dict['agent_state_dict']
            wm_state = {k[4:]: v for k, v in full_state.items() if k.startswith('_wm.')}
            behavior_state = {k[15:]: v for k, v in full_state.items() if k.startswith('_task_behavior.')}
            self._wm.load_state_dict(wm_state)
            self._task_behavior.load_state_dict(behavior_state)
        else:
            # Handle simple format
            self._wm.load_state_dict(state_dict['wm'])
            self._task_behavior.load_state_dict(state_dict['task_behavior'])
    
    def eval(self):
        self.training = False
        self._wm.eval()
        self._task_behavior.eval()


def create_render_env(config, env_id=0):
    """Create a single environment with rendering enabled"""
    return make_env(config, "eval", env_id)


def play_model(config):
    """Load model and play interactively with rendering"""
    
    # Setup directories
    logdir = pathlib.Path(config.logdir).expanduser() if config.logdir else pathlib.Path.cwd() / "logdir"
    checkpoint_path = logdir / "latest.pt"
    
    print(f"Looking for checkpoint in: {logdir}")
    
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("Make sure training has created a checkpoint file.")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Create environment with rendering
    print("Creating environment with rendering...")
    try:
        env = create_render_env(config)
        print(f"✅ Environment created: {type(env).__name__}")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space keys: {list(env.observation_space.spaces.keys())}")
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
        agent = SimpleAgent(obs_space, act_space, config)
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
        agent.load_state_dict(checkpoint)
        agent.eval()  # Set to evaluation mode
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Trying alternative checkpoint format...")
        try:
            # Try loading just the agent state dict
            agent.load_state_dict({"agent_state_dict": checkpoint["agent_state_dict"]})
            agent.eval()
            print(f"✅ Model loaded with alternative format!")
        except Exception as e2:
            print(f"❌ Failed with alternative format too: {e2}")
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
            
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                step += 1
                
                # Get action from agent
                with torch.no_grad():
                    action = agent(obs, training=False)
                
                # Extract action if it's a dictionary (for discrete action spaces)
                if isinstance(action, dict) and 'action' in action:
                    action = action['action']
                elif isinstance(action, dict):
                    # For other action formats, take the first value
                    action = next(iter(action.values()))
                
                # Step environment
                obs, reward, done, info = env.step(action)
                if callable(obs):
                    obs = obs()
                
                total_reward += reward
                
                # Print progress
                if step % 10 == 0 or done:
                    print(f"  Step {step:3d}: Reward {reward:6.3f}, Total: {total_reward:8.3f}")
                
                # Add small delay to make it watchable
                time.sleep(0.05)
                
                # Safety check
                if step > 1000:
                    print("  Episode too long, terminating...")
                    break
            
            print(f"Episode {episode} finished: {step} steps, {total_reward:.3f} total reward")
            
            # Ask if user wants to continue
            try:
                response = input("\nContinue? (y/n/Enter=yes): ").strip().lower()
                if response in ['n', 'no', 'q', 'quit']:
                    break
            except KeyboardInterrupt:
                break
    
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
    parser.add_argument("--configs", nargs="+", default=["defaults"], 
                       help="Configuration names to use (e.g., vertebrate)")
    parser.add_argument("--logdir", type=str, 
                       help="Directory containing the checkpoint (latest.pt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (default: cuda:0)")
    
    args = parser.parse_args()
    
    print(f"🎯 Loading configuration: {args.configs}")
    
    # Load configuration
    try:
        config = load_config(args.configs)
        
        # Override with command line args
        if args.logdir:
            config.logdir = args.logdir
        config.device = args.device
        
        print(f"✅ Configuration loaded")
        print(f"   Task: {config.task}")
        print(f"   Device: {config.device}")
        print(f"   Logdir: {config.logdir}")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Play the model
    play_model(config)


if __name__ == "__main__":
    main()