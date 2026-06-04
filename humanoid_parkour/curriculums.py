# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_vel_replay(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum with 80% normal + 20% random low-level replay to prevent forgetting."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    n_envs = env.scene.num_envs
    device = env.device

    # 1. Normal curriculum for 80% of envs
    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[:, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # 2. Randomly select 20% of envs for low-level replay
    replay_mask = torch.rand(n_envs, device=device) < 0.2
    replay_ids = replay_mask.nonzero(as_tuple=False).squeeze(-1)

    # 3. For replay envs: move DOWN to force revisit low levels
    move_down[replay_ids] = True
    move_up[replay_ids] = False

    terrain.update_env_origins(torch.arange(n_envs, device=device), move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_vel_adaptive(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Adaptive curriculum: normal advancement + fall-driven replay + anti-forgetting.

    Three components:
    1. 80%: normal curriculum (advance if walked far, retreat if insufficient progress)
    2. Adaptive: envs that barely moved (distance < 0.5m) are forced DOWN to easier levels
    3. Anti-forgetting: 10% uniform random replay to low levels
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    n_envs = env.scene.num_envs
    device = env.device

    # 1. Normal curriculum
    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[:, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # 2. Adaptive: envs that likely fell (distance < 0.5m) → force down
    fall_mask = distance < 0.5
    move_down[fall_mask] = True
    move_up[fall_mask] = False

    # 3. Anti-forgetting: 10% uniform random replay
    replay_mask = torch.rand(n_envs, device=device) < 0.1
    move_down[replay_mask] = True
    move_up[replay_mask] = False

    terrain.update_env_origins(torch.arange(n_envs, device=device), move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
