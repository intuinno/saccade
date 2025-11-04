"""
Custom parallel processing module using portal, inspired by embodied.driver
Compatible with tools.simulate and existing saccade environment patterns
"""

import time
import multiprocessing as mp
import cloudpickle
import numpy as np
import portal


class ParallelDriver:
    """
    Parallel environment driver using portal for process management.
    
    Inspired by embodied.Driver but designed to work with tools.simulate
    and maintain compatibility with saccade's callable environment pattern.
    """
    
    def __init__(self, make_env_fns, parallel=True):
        self.length = len(make_env_fns)
        self.parallel = parallel
        self.envs = []
        self.processes = []
        self.pipes = []
        
        if parallel and self.length > 1:
            print(f"🚀 Starting {self.length} parallel environments with portal")
            self._create_parallel_envs(make_env_fns)
        else:
            print(f"🔄 Creating {self.length} serial environments")
            self._create_serial_envs(make_env_fns)
    
    def _create_parallel_envs(self, make_env_fns):
        """Create parallel environments using portal processes"""
        # Create pipes for communication
        parent_pipes = []
        child_pipes = []
        for _ in range(self.length):
            parent_pipe, child_pipe = mp.Pipe()
            parent_pipes.append(parent_pipe)
            child_pipes.append(child_pipe)
        
        self.pipes = parent_pipes
        
        # Serialize environment constructors
        serialized_fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
        
        # Start worker processes using portal
        for i, (serialized_fn, child_pipe) in enumerate(zip(serialized_fns, child_pipes)):
            proc = portal.Process(
                self._env_worker,
                i, child_pipe, serialized_fn,
                start=True
            )
            self.processes.append(proc)
        
        # Create environment wrappers that work with tools.simulate
        for i in range(self.length):
            env_wrapper = ParallelEnvWrapper(i, self)
            self.envs.append(env_wrapper)
    
    def _create_serial_envs(self, make_env_fns):
        """Create serial environments for non-parallel execution"""
        self.envs = [fn() for fn in make_env_fns]
        for i, env in enumerate(self.envs):
            env.id = i
    
    @staticmethod
    def _env_worker(worker_id, pipe, serialized_fn):
        """
        Worker process function that runs one environment.
        
        This function runs in a separate process and handles commands
        from the main process via the pipe.
        """
        env = None
        try:
            print(f"Worker {worker_id}: Initializing environment...")
            
            # Deserialize and create environment
            env_fn = cloudpickle.loads(serialized_fn)
            env = env_fn()
            
            print(f"Worker {worker_id}: Environment created, entering command loop")
            
            while True:
                try:
                    # Wait for command from main process
                    if not pipe.poll(timeout=1.0):
                        continue
                    
                    cmd = pipe.recv()
                    
                    if cmd[0] == 'reset':
                        obs = env.reset()
                        # Handle saccade's callable pattern
                        if callable(obs):
                            obs = obs()
                        pipe.send(('success', obs))
                        
                    elif cmd[0] == 'step':
                        action = cmd[1]
                        result = env.step(action)
                        # Handle saccade's callable pattern
                        if callable(result):
                            result = result()
                        pipe.send(('success', result))
                        
                    elif cmd[0] == 'getattr':
                        attr_name = cmd[1]
                        try:
                            attr_value = getattr(env, attr_name)
                            pipe.send(('success', attr_value))
                        except AttributeError:
                            pipe.send(('error', f"Attribute '{attr_name}' not found"))
                            
                    elif cmd[0] == 'close':
                        print(f"Worker {worker_id}: Closing environment")
                        if hasattr(env, 'close'):
                            env.close()
                        pipe.send(('success', None))
                        break
                        
                    else:
                        pipe.send(('error', f"Unknown command: {cmd[0]}"))
                        
                except EOFError:
                    print(f"Worker {worker_id}: Pipe closed, shutting down")
                    break
                except Exception as e:
                    print(f"Worker {worker_id}: Command error: {e}")
                    pipe.send(('error', str(e)))
                    
        except Exception as e:
            print(f"Worker {worker_id}: Initialization error: {e}")
        finally:
            try:
                if env and hasattr(env, 'close'):
                    env.close()
                if pipe:
                    pipe.close()
            except:
                pass
    
    def close(self):
        """Close all environments and cleanup processes"""
        if self.parallel and self.processes:
            print("🛑 Closing parallel environments...")
            
            # Send close commands to all workers
            for i, pipe in enumerate(self.pipes):
                try:
                    pipe.send(('close',))
                    # Wait for confirmation
                    if pipe.poll(timeout=2.0):
                        pipe.recv()
                except:
                    pass
            
            # Wait for processes to terminate gracefully
            for i, proc in enumerate(self.processes):
                try:
                    proc.join(timeout=3.0)
                    if proc.is_alive():
                        print(f"Force killing worker {i}")
                        proc.kill()
                except:
                    pass
            
            print("✅ All parallel environments closed")
        
        # Close serial environments
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def __del__(self):
        """Cleanup when driver is deleted"""
        try:
            self.close()
        except:
            pass


class ParallelEnvWrapper:
    """
    Wrapper that makes a parallel environment slot look like an individual environment.
    
    This maintains full compatibility with tools.simulate expectations:
    - Has reset() and step() methods that return callables
    - Has action_space, observation_space, and id attributes
    - Handles the saccade pattern seamlessly
    """
    
    def __init__(self, env_id, driver):
        self.env_id = env_id
        self.driver = driver
        self.id = env_id
        
        # Try to get environment attributes from the worker process
        if driver.parallel:
            self._fetch_attributes_from_worker()
    
    def _fetch_attributes_from_worker(self):
        """Get environment attributes from worker process"""
        try:
            # Get action_space
            self.driver.pipes[self.env_id].send(('getattr', 'action_space'))
            status, result = self.driver.pipes[self.env_id].recv()
            if status == 'success':
                self.action_space = result
            
            # Get observation_space
            self.driver.pipes[self.env_id].send(('getattr', 'observation_space'))
            status, result = self.driver.pipes[self.env_id].recv()
            if status == 'success':
                self.observation_space = result
            
            # Set compatibility attributes (some environments use these names)
            self.act_space = getattr(self, 'action_space', None)
            self.obs_space = getattr(self, 'observation_space', None)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not get attributes from worker {self.env_id}: {e}")
    
    def reset(self):
        """
        Reset environment - returns callable for saccade compatibility.
        
        This method returns a callable that, when called, performs the actual reset.
        This matches the pattern expected by tools.simulate.
        """
        def _reset():
            if self.driver.parallel:
                self.driver.pipes[self.env_id].send(('reset',))
                status, result = self.driver.pipes[self.env_id].recv()
                if status == 'error':
                    raise RuntimeError(f"Reset error in env {self.env_id}: {result}")
                return result
            else:
                # Serial execution - call the environment directly
                obs = self.driver.envs[self.env_id].reset()
                if callable(obs):
                    obs = obs()
                return obs
        
        return _reset
    
    def step(self, action):
        """
        Step environment - returns callable for saccade compatibility.
        
        This method returns a callable that, when called, performs the actual step.
        This matches the pattern expected by tools.simulate.
        """
        def _step():
            if self.driver.parallel:
                self.driver.pipes[self.env_id].send(('step', action))
                status, result = self.driver.pipes[self.env_id].recv()
                if status == 'error':
                    raise RuntimeError(f"Step error in env {self.env_id}: {result}")
                return result
            else:
                # Serial execution - call the environment directly
                result = self.driver.envs[self.env_id].step(action)
                if callable(result):
                    result = result()
                return result
        
        return _step
    
    def close(self):
        """Close handled by the driver"""
        pass


def create_parallel_envs(make_env_fns, parallel=True):
    """
    Convenience function to create parallel environments.
    
    Args:
        make_env_fns: List of environment constructor functions
        parallel: Whether to use parallel processing
        
    Returns:
        List of environment wrappers compatible with tools.simulate
    """
    driver = ParallelDriver(make_env_fns, parallel=parallel)
    return driver.envs, driver


# Test function to verify the parallel mechanism works
def test_parallel_envs():
    """Test function to verify parallel environments work correctly"""
    print("🧪 Testing parallel environments...")
    
    def make_dummy_env():
        """Create a simple dummy environment for testing"""
        import gym
        return gym.make('CartPole-v1')
    
    # Test with 2 environments
    make_env_fns = [make_dummy_env, make_dummy_env]
    envs, driver = create_parallel_envs(make_env_fns, parallel=True)
    
    try:
        # Test reset
        obs_list = []
        for env in envs:
            obs = env.reset()()  # Call the callable
            obs_list.append(obs)
        print(f"✅ Reset successful: got {len(obs_list)} observations")
        
        # Test step
        for env in envs:
            action = env.action_space.sample()
            result = env.step(action)()  # Call the callable
            print(f"✅ Step successful: {type(result)}")
        
        print("🎉 All tests passed!")
        
    finally:
        driver.close()


if __name__ == "__main__":
    test_parallel_envs()