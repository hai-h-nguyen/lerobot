"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"`
```
"""

from pathlib import Path

import gym_peg_insertion
import gymnasium as gym
import imageio
import numpy as np
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

peg_type = 'square'

# Create a directory to store the video of the evaluation
output_directory = Path(f"output/eval/peg-insertion-{peg_type}-w-ft")
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
pretrained_policy_path = Path(f"output/peg-insertion-{peg_type}-front-view-windowed-ft/checkpoints/last/pretrained_model")
# OR a path to a local outputs/train folder.

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, map_location=device)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    f"gym_peg_insertion/{peg_type}-v0",
    obs_type="pixels_agent_pos_w_ft",
    render_mode="rgb_array",
    max_episode_steps=300,
    camera_widths=(256, 256, 1024),
    camera_heights=(256, 256, 1024),
)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)
print(env.action_space)

num_test_episodes = 10

# Reset the policy and environments to prepare for rollout
num_successes = 0

for episode_idx in range(num_test_episodes):
    policy.reset()
    numpy_observation, info = env.reset()

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())  # Capture the initial frame from the environment
    # frames.append(np.transpose(numpy_observation["cam_front_view"], (1, 2, 0)))

    step = 0
    done = False
    while not done:
        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"].copy())  # Extract the state from the numpy observation
        image_front_view = torch.from_numpy(numpy_observation["pixels"]["front"].copy())
        image_eye_in_hand = torch.from_numpy(numpy_observation["pixels"]["eye_in_hand"].copy())
        image_side_view = torch.from_numpy(numpy_observation["pixels"]["side"].copy())

        image_front_view = image_front_view.permute(2, 0, 1)  # Change from HWC to CHW format for PyTorch
        image_eye_in_hand = image_eye_in_hand.permute(2, 0, 1)  # Change from HWC to CHW format for PyTorch
        image_side_view = image_side_view.permute(2, 0, 1)  # Change from HWC to CHW format for PyTorch

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        image_front_view = image_front_view.to(torch.float32) / 255
        image_eye_in_hand = image_eye_in_hand.to(torch.float32) / 255
        image_side_view = image_side_view.to(torch.float32) / 255

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image_front_view = image_front_view.to(device, non_blocking=True)
        image_eye_in_hand = image_eye_in_hand.to(device, non_blocking=True)
        image_side_view = image_side_view.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image_front_view = image_front_view.unsqueeze(0)
        image_eye_in_hand = image_eye_in_hand.unsqueeze(0)
        image_side_view = image_side_view.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.front": image_front_view,
            "observation.images.eye_in_hand": image_eye_in_hand,
            "observation.images.side": image_side_view,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        # print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    mark_success = ""
    if terminated:
        print("Success!")
        mark_success = "Success"
        num_successes += 1
    else:
        mark_success = "Failure"
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = f"{output_directory}/rollout_{episode_idx}_{mark_success}.mp4"
    imageio.mimsave(video_path, np.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")

print(num_successes, num_test_episodes)
