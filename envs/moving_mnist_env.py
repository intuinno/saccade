import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self) -> None:
        pass

    # --- Private methods ---

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _reset(self):
        pass

    # --- Public methods ---

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
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

    The observation returned by the environment is now a dictionary containing:
        - 'central': Tensor of shape [batch_size, patch_size, patch_size], the image patches under the agent's position.
        - 'peripheral': Tensor of shape [batch_size, 8, 8], the 8x8 downsampled version of the whole Moving MNIST image.
        - 'loc': Tensor of shape [batch_size,], the location of the agent position in number
    """

    def __init__(self, num_digits=2, image_size=64, batch_size=32, device="cuda"):
        super(MovingMNISTEnv, self).__init__()
        self.num_digits = num_digits  # Number of moving digits
        self.image_size = image_size  # Must be divisible by grid_size
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
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    # --- Private methods ---

    def _get_obs(self):
        """
        Extracts the observation from the current state based on the agent's position.
        Returns a dictionary containing:
            - 'patch': Tensor of shape [batch_size, patch_size, patch_size]
            - 'downsampled_frame': Tensor of shape [batch_size, 8, 8]
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

        # Downsample frames to 8x8
        downsampled_frames = F.interpolate(
            frames.unsqueeze(1), size=(8, 8), mode="bilinear", align_corners=False
        ).squeeze(
            1
        )  # Shape: [batch_size, 8, 8]

        # loc is a location of the focus
        loc = self.agent_pos[:, 0] + self.agent_pos[:, 1]
        loc = F.one_hot(loc, num_classes=16)
        patches = torch.flatten(patches, start_dim=1)
        downsampled_frames = torch.flatten(downsampled_frames, start_dim=1)
        # Return as a dictionary
        observation = {
            "central": patches,  # Shape: [batch_size, patch_size, patch_size]
            "peripheral": downsampled_frames,  # Shape: [batch_size, 8, 8]
            "loc": loc,
            "GT": frames,
        }

        return observation

    def _reset(self):
        """
        Resets the environment and generates a new batch of sequences.
        """
        self.current_step = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

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

        return observation

    def _reset_if_done(self):
        """
        Resets the environments where done is True.
        """
        done = self.done
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
            # Reset done flags
            self.done[indices] = False
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

        # Initialize frames tensor
        frames = torch.zeros(
            self.batch_size, self.image_size, self.image_size, device=self.device
        )

        # Get integer positions
        x = self.pos[:, :, 0].long()  # Shape: [batch_size, num_digits]
        y = self.pos[:, :, 1].long()  # Shape: [batch_size, num_digits]

        # Flatten batch and digit dimensions
        batch_digits = self.batch_size * self.num_digits
        digits_flat = self.digits.view(batch_digits, 28, 28)
        x_flat = x.view(batch_digits)
        y_flat = y.view(batch_digits)

        # Create coordinate grids for the digits
        grid_y, grid_x = torch.meshgrid(
            torch.arange(28, device=self.device), torch.arange(28, device=self.device)
        )  # Shape: [28, 28]

        # Expand grids to match batch size
        grid_x = (
            grid_x.unsqueeze(0) + x_flat[:, None, None]
        )  # Shape: [batch_digits, 28, 28]
        grid_y = (
            grid_y.unsqueeze(0) + y_flat[:, None, None]
        )  # Shape: [batch_digits, 28, 28]

        # Flatten the grids and digits
        grid_x_flat = grid_x.reshape(-1)  # Shape: [batch_digits * 28 * 28]
        grid_y_flat = grid_y.reshape(-1)
        digits_flat_flat = digits_flat.reshape(-1)

        # Create batch indices
        batch_indices = (
            torch.arange(self.batch_size, device=self.device)
            .unsqueeze(1)
            .repeat(1, self.num_digits)
            .view(-1)
        )
        batch_indices = batch_indices[:, None, None].expand(batch_digits, 28, 28)
        batch_indices_flat = batch_indices.reshape(-1)

        # Mask to keep indices within bounds
        mask = (
            (grid_x_flat >= 0)
            & (grid_x_flat < self.image_size)
            & (grid_y_flat >= 0)
            & (grid_y_flat < self.image_size)
        )

        # Apply mask to indices and digits
        batch_indices_flat = batch_indices_flat[mask]
        grid_x_flat = grid_x_flat[mask]
        grid_y_flat = grid_y_flat[mask]
        digits_flat_flat = digits_flat_flat[mask]

        # Use index_put_ with accumulate=True to add digit pixels to frames
        frames.index_put_(
            (batch_indices_flat, grid_y_flat, grid_x_flat),
            digits_flat_flat,
            accumulate=True,
        )

        # Clip pixel values to [0, 1]
        frames = torch.clamp(frames, 0, 1)

        # Update frames in self.frames
        self.frames = frames

    # --- Public methods ---

    def reset(self):
        """
        Public method to reset the environment.

        Returns:
            observation: Dictionary containing:
                - 'patch': Tensor of shape [batch_size, patch_size, patch_size]
                - 'downsampled_frame': Tensor of shape [batch_size, 8, 8]
        """
        observation = self._reset()
        return observation

    def step(self, action):
        """
        Steps through the environment.

        Args:
            action: Dictionary with keys:
                'delta_x': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in X.
                'delta_y': Tensor of shape [batch_size, 7], one-hot vectors representing shifts from -3 to +3 in Y.
                'digits': Tensor of shape [batch_size, 2], integers between 0 and 9 (guessed digits).
                'guess': Tensor of shape [batch_size], boolean flags (0 or 1).

        Returns:
            observation: Dictionary containing:
                - 'patch': Tensor of shape [batch_size, patch_size, patch_size]
                - 'downsampled_frame': Tensor of shape [batch_size, 8, 8]
            reward: Tensor of shape [batch_size], rewards for each environment in the batch.
            done: Tensor of shape [batch_size], boolean flags indicating if the episode is done.
        """
        # Extract components from action
        delta_x_action = action["delta_x"]  # Shape: [batch_size, 7]
        delta_y_action = action["delta_y"]  # Shape: [batch_size, 7]
        guess_digits = action["digits"]  # Shape: [batch_size, self.num_digits]
        guess_flag = action["guess"].squeeze()  # Shape: [batch_size]

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
        observation = (
            self._get_obs()
        )  # Dictionary containing 'patch' and 'downsampled_frame'

        # Initialize reward tensor
        reward = torch.zeros(self.batch_size, device=self.device)

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
            self.done[guess_indices[correct_guess]] = True
            reward[guess_indices[~correct_guess]] = -10.0
            # If incorrect guess, environment continues (done=False)

        # Increment current_step for all environments
        self.current_step += 1

        # Copy of the done flags before resetting
        done_flags = self.done.clone()

        # Reset environments that are done
        if self.done.any():
            self._reset_if_done()

        # Generate next frame for all environments
        self._generate_frame()

        # Return observation, reward, done_flags
        return observation, reward, done_flags

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
