import pygame
import torch
import numpy as np
from moving_mnist_env import (
    MovingMNISTEnv,
)  # Ensure this file is in your working directory


def main():
    # Initialize pygame
    pygame.init()

    # Environment parameters
    num_digits = 2
    image_size = 64
    num_frames = 100  # Number of frames in each sequence
    batch_size = 3  # Display 6 environments
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the MovingMNISTEnv environment
    env = MovingMNISTEnv(
        num_digits=num_digits,
        image_size=image_size,
        num_frames=num_frames,
        batch_size=batch_size,
        device=device,
    )

    # Reset the environment
    observation = env.reset()
    # Extract 'patch' and 'downsampled_frame' from the observation
    patches = observation["patch"]  # Shape: [batch_size, patch_size, patch_size]
    downsampled_frames = observation["downsampled_frame"]  # Shape: [batch_size, 8, 8]

    # Display parameters
    grid_rows = 2
    grid_cols = 3
    base_window_width = grid_cols * image_size
    base_window_height = (
        grid_rows * image_size + 100
    )  # Extra space for text and FPS slider
    window_width, window_height = base_window_width, base_window_height
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Moving MNIST Environment - Batch of 6")

    # Font for displaying text
    font_size = 6
    font = pygame.font.SysFont("Arial", font_size)

    # Clock object to control the frame rate
    clock = pygame.time.Clock()
    fps = 5  # Default FPS

    # Variables for FPS slider
    slider_width = 200
    slider_height = 20
    slider_color = (200, 200, 200)
    knob_color = (100, 100, 100)
    knob_radius = 10
    dragging = False

    # FPS range
    min_fps = 1
    max_fps = 30

    # Calculate initial knob position
    slider_x = (window_width - slider_width) // 2
    slider_y = window_height - 40  # Position at the bottom of the window
    knob_x = slider_x + int((fps - min_fps) / (max_fps - min_fps) * slider_width)
    knob_y = slider_y + slider_height // 2

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Update window size
                window_width, window_height = event.w, event.h
                screen = pygame.display.set_mode(
                    (window_width, window_height), pygame.RESIZABLE
                )
                # Update slider position
                slider_x = (window_width - slider_width) // 2
                slider_y = window_height - 40
                knob_y = slider_y + slider_height // 2
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if mouse is over the knob
                    mouse_x, mouse_y = event.pos
                    if (
                        knob_x - knob_radius <= mouse_x <= knob_x + knob_radius
                        and knob_y - knob_radius <= mouse_y <= knob_y + knob_radius
                    ):
                        dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    # Update knob position based on mouse x position
                    mouse_x, mouse_y = event.pos
                    knob_x = max(slider_x, min(mouse_x, slider_x + slider_width))
                    # Update FPS based on knob position
                    fps = min_fps + (knob_x - slider_x) / slider_width * (
                        max_fps - min_fps
                    )
                    fps = max(min_fps, min(int(fps), max_fps))

        # Sample random actions
        actions = {
            "delta_x": torch.zeros(env.batch_size, 7, device=device),
            "delta_y": torch.zeros(env.batch_size, 7, device=device),
            "digits": torch.randint(
                0, 10, (env.batch_size, env.num_digits), device=device
            ),
            "guess": torch.randint(0, 2, (env.batch_size,), device=device),
        }

        # Randomly select a delta for each environment
        delta_indices_x = torch.randint(0, 7, (env.batch_size,), device=device)
        delta_indices_y = torch.randint(0, 7, (env.batch_size,), device=device)
        actions["delta_x"][
            torch.arange(env.batch_size), delta_indices_x
        ] = 1  # Set one random delta in X
        actions["delta_y"][
            torch.arange(env.batch_size), delta_indices_y
        ] = 1  # Set one random delta in Y

        # Step the environment
        observation, reward, done = env.step(actions)
        # Extract 'patch' and 'downsampled_frame' from the observation
        patches = observation["patch"]  # Shape: [batch_size, patch_size, patch_size]
        downsampled_frames = observation[
            "downsampled_frame"
        ]  # Shape: [batch_size, 8, 8]

        # Clear the screen
        screen.fill((0, 0, 0))

        # Calculate scaling factors
        scale_x = window_width / base_window_width
        scale_y = (window_height - 100) / (base_window_height - 100)
        scale = min(scale_x, scale_y)

        # Adjusted image size
        scaled_image_size = int(image_size * scale)
        scaled_patch_size = int(env.patch_size * scale)
        scaled_font_size = int(font_size * scale)
        scaled_text_offset = int(5 * scale)

        # Update font size
        font = pygame.font.SysFont("Arial", max(scaled_font_size, 12))

        # Loop over each environment in the batch
        for i in range(batch_size):
            # Calculate position in the grid
            row = i // grid_cols
            col = i % grid_cols
            x_offset = int(col * scaled_image_size)
            y_offset = int(row * scaled_image_size)

            # Get the frame to display
            frame = env.frames[i].cpu().numpy()  # Shape: (image_size, image_size)

            # Resize the frame
            frame_rgb = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            frame_surface = pygame.surfarray.make_surface(np.uint8(frame_rgb * 255))
            frame_surface = pygame.transform.scale(
                frame_surface, (scaled_image_size, scaled_image_size)
            )

            # Blit the frame onto the screen
            screen.blit(frame_surface, (x_offset, y_offset))

            # Draw the grid and highlight the agent's current cell
            grid_color = (255, 255, 255)  # White grid lines
            agent_color = (0, 255, 0)  # Green highlight

            for gx in range(env.grid_size):
                for gy in range(env.grid_size):
                    rect = pygame.Rect(
                        x_offset + gx * scaled_patch_size,
                        y_offset + gy * scaled_patch_size,
                        scaled_patch_size,
                        scaled_patch_size,
                    )
                    pygame.draw.rect(screen, grid_color, rect, max(1, int(1 * scale)))

            # Highlight the agent's current cell
            agent_pos = env.agent_pos[i]
            agent_rect = pygame.Rect(
                x_offset + agent_pos[0] * scaled_patch_size,
                y_offset + agent_pos[1] * scaled_patch_size,
                scaled_patch_size,
                scaled_patch_size,
            )
            pygame.draw.rect(screen, agent_color, agent_rect, max(2, int(3 * scale)))

            # Display the patch under the agent
            # Let's display the patch at the top-left corner of each frame
            scaled_patch_display_size = int(scaled_patch_size * 0.5)
            patch_image = patches[i].cpu().numpy()
            patch_image_rgb = np.repeat(patch_image[:, :, np.newaxis], 3, axis=2)
            patch_surface = pygame.surfarray.make_surface(
                np.uint8(patch_image_rgb * 255)
            )
            patch_surface = pygame.transform.scale(
                patch_surface, (scaled_patch_display_size, scaled_patch_display_size)
            )
            screen.blit(patch_surface, (x_offset, y_offset))

            # Optionally, display the downsampled frame (commented out here)
            downsampled_image = downsampled_frames[i].cpu().numpy()
            downsampled_image_rgb = np.repeat(
                downsampled_image[:, :, np.newaxis], 3, axis=2
            )
            downsampled_surface = pygame.surfarray.make_surface(
                np.uint8(downsampled_image_rgb * 255)
            )
            downsampled_surface = pygame.transform.scale(
                downsampled_surface,
                (scaled_patch_display_size, scaled_patch_display_size),
            )
            screen.blit(
                downsampled_surface,
                (x_offset + scaled_image_size - scaled_patch_display_size, y_offset),
            )

            # Display the random guess
            guess_digits = actions["digits"][i].cpu().numpy()
            guess_text = f"Guess: {guess_digits[0]}, {guess_digits[1]}"
            text_surface = font.render(guess_text, True, (255, 255, 255))
            text_x = x_offset
            text_y = (
                y_offset + scaled_image_size + scaled_text_offset
            )  # Position below the frame

            # Ensure text does not overlap with FPS slider or other text
            max_text_y = slider_y - text_surface.get_height() - scaled_text_offset
            if text_y > max_text_y:
                text_y = max_text_y

            screen.blit(text_surface, (text_x, text_y))

        # Draw the FPS slider
        # Draw the slider line
        pygame.draw.rect(
            screen, slider_color, (slider_x, slider_y, slider_width, slider_height)
        )
        # Draw the knob
        pygame.draw.circle(screen, knob_color, (knob_x, knob_y), knob_radius)
        # Draw FPS text
        fps_text = f"FPS: {fps}"
        fps_text_surface = font.render(fps_text, True, (255, 255, 255))
        fps_text_rect = fps_text_surface.get_rect(
            center=(window_width // 2, slider_y - 20)
        )
        screen.blit(fps_text_surface, fps_text_rect)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(fps)

        # If all environments are done, reset
        if done.all():
            observation = env.reset()
            patches = observation["patch"]
            downsampled_frames = observation["downsampled_frame"]

    # Clean up
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
