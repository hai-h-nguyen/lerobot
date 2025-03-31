#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib

import gymnasium as gym
import pytest
import torch
from gymnasium.utils.env_checker import check_env

import lerobot
from lerobot.common.envs.factory import make_env, make_env_config
from lerobot.common.envs.utils import preprocess_observation

# from .utils import require_env

OBS_TYPES = ["pixels_agent_pos_no_ft", "pixels_agent_pos_w_ft"]


def test_env(env_name, env_task, obs_type):
    package_name = "gym_peg_insertion"
    importlib.import_module(package_name)
    env = gym.make(f"{package_name}/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_factory(env_name):
    cfg = make_env_config(env_name)
    env = make_env(cfg, n_envs=1)
    obs, _ = env.reset()
    obs = preprocess_observation(obs)

    # test image keys are float32 in range [0,1]
    for key in obs:
        if "image" not in key:
            continue
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    env.close()


test_env("env_name", "square-v0", "pixels_agent_pos_no_ft")
