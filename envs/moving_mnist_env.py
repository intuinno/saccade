import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self) -> None:
        pass

    # --- Private methods ---

    @abstractmethod
    def _get_obs(self, state):
        pass

    @abstractmethod
    def _reset(self, key):
        pass

    @abstractmethod
    def _reset_if_done(self, env_state, done):
        pass

    # --- Public methods ---

    @abstractmethod
    def reset(self, key):
        pass

    @abstractmethod
    def step(self, env_state, action):
        pass


class MovingMNISTEnv(BaseEnv):
    """
    A vectorized Moving MNIST environment where the action includes delta movements
    in X and Y directions. The agent's position is updated based on the deltas.

    The action includes:
        - 'delta_x': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in X.
        - 'delta_y': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in Y.
        - 'digits': Tensor of shape [batch_size, 2], integers between 0 and 9 (guessed digits).
        - 'guess': Tensor of shape [batch_size], boolean flags (0 or 1).
    """

    def __init__(
        self, num_digits=2, image_size=64, num_frames=20, batch_size=32, device="cuda"
    ):
        super(MovingMNISTEnv, self).__init__()
        self.num_digits = num_digits  # Number of moving digits (set to 2)
        self.image_size = image_size  # Must be divisible by grid_size
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.device = device

        # Grid parameters
        self.grid_size = 4  # 4x4 grid
        self.patch_size = (
            self.image_size // self.grid_size
        )  # Size of each patch (16x16)

        # Load MNIST dataset
        self.mnist = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        self.data = self.mnist.data.float() / 255.0  # Normalize to [0, 1]
        self.targets = self.mnist.targets  # Labels

        # Preload digits and labels to GPU
        self.dataset_digits = self.data.to(self.device)
        self.dataset_labels = self.targets.to(self.device)

        # Internal state
        self.current_step = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.frames = None  # Current frames for each environment
        self.pos = None
        self.velocity = None
        self.digits = None  # Digit images in the environment
        self.digit_labels = None  # Actual labels of the digits

        # Agent positions on the grid (initialize in _reset)
        self.agent_pos = None  # Shape: [batch_size, 2], where each position is [x, y]

    # --- Private methods ---

    def _get_obs(self):
        """
        Extracts the observation from the current state based on the agent's position.
        """
        frames = self.frames

        # Get agent positions
        grid_x = self.agent_pos[:, 0]
        grid_y = self.agent_pos[:, 1]

        # Extract patches based on the agent's position
        patches = torch.zeros(
            self.batch_size, self.patch_size, self.patch_size, device=self.device
        )

        for b in range(self.batch_size):
            x_start = grid_x[b] * self.patch_size
            y_start = grid_y[b] * self.patch_size
            patches[b] = frames[
                b,
                y_start : y_start + self.patch_size,
                x_start : x_start + self.patch_size,
            ]

        return patches  # Shape: [batch_size, patch_size, patch_size]

    def _reset(self, key=None):
        """
        Resets the environment and generates a new batch of sequences.
        """
        self.current_step = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

        # Randomly select digits for each sequence
        indices = torch.randint(
            0,
            len(self.dataset_digits),
            (self.batch_size, self.num_digits),
            device=self.device,
        )
        digits = self.dataset_digits[indices]  # Shape: [batch_size, num_digits, 28, 28]
        digit_labels = self.dataset_labels[indices]  # Shape: [batch_size, num_digits]

        # Initialize positions and velocities
        pos = torch.rand(self.batch_size, self.num_digits, 2, device=self.device) * (
            self.image_size - 28
        )
        theta = (
            torch.rand(self.batch_size, self.num_digits, device=self.device) * 2 * np.pi
        )
        velocity = (
            torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * 2
        )  # Speed factor

        self.pos = pos
        self.velocity = velocity
        self.digits = digits  # Images of digits
        self.digit_labels = digit_labels  # Actual labels of the digits

        # Initialize frames tensor
        self.frames = torch.zeros(
            self.batch_size, self.image_size, self.image_size, device=self.device
        )

        # Initialize agent positions at the center of the grid
        self.agent_pos = torch.full(
            (self.batch_size, 2),
            self.grid_size // 2,
            device=self.device,
            dtype=torch.long,
        )

        # Generate the first frame
        self._generate_frame()

        # Initial observation
        observation = self._get_obs()  # Get the initial observation

        # Initial state
        env_state = {
            "frames": self.frames,
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            "agent_pos": self.agent_pos,
        }

        return env_state, observation

    def _reset_if_done(self, env_state, done):
        """
        Resets the environments where done is True.
        """
        if done.any():
            indices = done.nonzero(as_tuple=True)[0]
            # Reset current step for the done environments
            self.current_step[indices] = 0
            # Reinitialize positions and velocities for the done environments
            self.pos[indices] = torch.rand(
                len(indices), self.num_digits, 2, device=self.device
            ) * (self.image_size - 28)
            theta = (
                torch.rand(len(indices), self.num_digits, device=self.device)
                * 2
                * np.pi
            )
            self.velocity[indices] = (
                torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * 2
            )  # Speed factor
            # Reinitialize digits and labels
            idx_digits = torch.randint(
                0,
                len(self.dataset_digits),
                (len(indices), self.num_digits),
                device=self.device,
            )
            self.digits[indices] = self.dataset_digits[idx_digits]
            self.digit_labels[indices] = self.dataset_labels[idx_digits]
            # Reset agent positions to the center
            self.agent_pos[indices] = torch.full(
                (len(indices), 2),
                self.grid_size // 2,
                device=self.device,
                dtype=torch.long,
            )
            # Do not generate frames here; frames will be generated for all environments in _generate_frame()

    def _generate_frame(self):
        """
        Generates the frame at the current time step for all environments.
        """
        # Update positions
        self.pos += self.velocity

        # Bounce off the edges
        over = (self.pos < 0) | (self.pos > self.image_size - 28)
        self.velocity[over] *= -1
        self.pos = torch.clamp(self.pos, 0, self.image_size - 28)

        # Create frames for all environments
        frames = torch.zeros(
            self.batch_size, self.image_size, self.image_size, device=self.device
        )

        x = self.pos[:, :, 0].long()
        y = self.pos[:, :, 1].long()

        for i in range(self.num_digits):
            for b in range(self.batch_size):
                frames[
                    b, y[b, i] : y[b, i] + 28, x[b, i] : x[b, i] + 28
                ] += self.digits[b, i]

        # Clip pixel values to [0, 1]
        frames = torch.clamp(frames, 0, 1)

        # Update frames in self.frames
        self.frames = frames

    # --- Public methods ---

    def reset(self, key=None):
        """
        Public method to reset the environment.
        """
        env_state, observation = self._reset(key)
        return env_state, observation

    def step(self, env_state, action):
        """
        Steps through the environment.

        Args:
            env_state: Dictionary containing the current state of the environment.
            action: Dictionary with keys:
                'delta_x': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in X.
                'delta_y': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in Y.
                'digits': Tensor of shape [batch_size, 2], integers between 0 and 9 (guessed digits).
                'guess': Tensor of shape [batch_size], boolean flags (0 or 1).

        Returns:
            env_state: Updated environment state.
            observation: Tensor of shape [batch_size, patch_size, patch_size], image patches under the agent's position.
            reward: Tensor of shape [batch_size], rewards for each environment in the batch.
            done: Tensor of shape [batch_size], boolean flags indicating if the episode is done.
        """
        # Extract components from action
        delta_x_action = action["delta_x"]  # Shape: [batch_size, 7]
        delta_y_action = action["delta_y"]  # Shape: [batch_size, 7]
        guess_digits = action["digits"]  # Shape: [batch_size, num_digits]
        guess_flag = action["guess"]  # Shape: [batch_size]

        if isinstance(delta_x_action, np.ndarray):
            delta_x_action = torch.tensor(delta_x_action, device=self.device)
        if isinstance(delta_y_action, np.ndarray):
            delta_y_action = torch.tensor(delta_y_action, device=self.device)
        if isinstance(guess_digits, np.ndarray):
            guess_digits = torch.tensor(guess_digits, device=self.device)
        if isinstance(guess_flag, np.ndarray):
            guess_flag = torch.tensor(guess_flag, device=self.device)

        # Verify action shapes
        assert delta_x_action.shape == (
            self.batch_size,
            7,
        ), f"Expected delta_x_action shape ({self.batch_size}, 7), got {delta_x_action.shape}"
        assert delta_y_action.shape == (
            self.batch_size,
            7,
        ), f"Expected delta_y_action shape ({self.batch_size}, 7), got {delta_y_action.shape}"
        assert guess_digits.shape == (
            self.batch_size,
            self.num_digits,
        ), f"Expected guess_digits shape ({self.batch_size}, {self.num_digits}), got {guess_digits.shape}"
        assert guess_flag.shape == (
            self.batch_size,
        ), f"Expected guess_flag shape ({self.batch_size},), got {guess_flag.shape}"

        # Map one-hot vectors to deltas (-3 to +3)
        delta_values = torch.arange(-3, 4, device=self.device)
        delta_x = (
            (delta_x_action * delta_values).sum(dim=1).long()
        )  # Shape: [batch_size]
        delta_y = (
            (delta_y_action * delta_values).sum(dim=1).long()
        )  # Shape: [batch_size]

        # Update agent positions
        self.agent_pos[:, 0] += delta_x
        self.agent_pos[:, 1] += delta_y

        # Ensure agent positions are within the grid boundaries
        self.agent_pos[:, 0] = torch.clamp(self.agent_pos[:, 0], 0, self.grid_size - 1)
        self.agent_pos[:, 1] = torch.clamp(self.agent_pos[:, 1], 0, self.grid_size - 1)

        # Get observation based on updated agent positions
        observation = self._get_obs()  # Shape: [batch_size, patch_size, patch_size]

        # Initialize reward and done tensors
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Process 'guess' flag
        guess_flag = guess_flag.bool()  # Convert to boolean tensor

        if guess_flag.any():
            # Environments where guess_flag is True
            guess_indices = guess_flag.nonzero(as_tuple=True)[0]
            # Sort the guessed digits and actual digits
            sorted_guess_digits, _ = torch.sort(guess_digits[guess_indices], dim=1)
            sorted_actual_digits, _ = torch.sort(
                self.digit_labels[guess_indices], dim=1
            )
            # Compare sorted digits
            correct_guess = torch.all(
                sorted_guess_digits == sorted_actual_digits, dim=1
            )
            # Assign rewards and done flags
            reward[guess_indices[correct_guess]] = 100.0
            done[guess_indices[correct_guess]] = True
            reward[guess_indices[~correct_guess]] = -10.0
            # If incorrect guess, environment continues (done=False)

        # Increment current_step for all environments
        self.current_step += 1

        # Check if episodes should end due to reaching num_frames
        reached_max_steps = self.current_step >= self.num_frames
        done = done | reached_max_steps

        # Reset environments that are done
        if done.any():
            self._reset_if_done(env_state, done)

        # Generate next frame for all environments
        self._generate_frame()
        env_state["frames"] = self.frames
        env_state["done"] = done
        env_state["agent_pos"] = self.agent_pos

        return env_state, observation, reward, done

    def render(self, idx=0):
        """
        Renders the environment for a specific sequence in the batch.

        Args:
            idx: Index of the sequence in the batch to render.
        """
        import matplotlib.pyplot as plt

        if self.frames is None:
            print("Environment not reset. Call env.reset() before rendering.")
            return

        frame = self.frames[idx].cpu().numpy()  # Frame at index idx

        plt.imshow(frame, cmap="gray")
        plt.title(f"Environment {idx}")
        plt.show()

    def close(self):
        """
        Cleans up the environment.
        """
        pass
