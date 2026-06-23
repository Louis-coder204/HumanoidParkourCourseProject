#!/usr/bin/env python
"""Record G1 locomotion video on specific PARKOUR terrain types.

Usage:
    python record_terrain_video.py --terrain_type gaps --difficulty easy --out videos/gaps_easy.mp4
    python record_terrain_video.py --terrain_type stairs --difficulty hard --out videos/stairs_hard.mp4

Available terrain types: gaps, stairs, stones, boxes
Available difficulties: easy, hard

Checkpoint: HikingSafetyLite model_2999.pt (best PARKOUR model, 89.3% avg success)
"""

import argparse
import sys
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--terrain_type", type=str, required=True,
                    choices=["gaps", "stairs", "stones", "boxes"])
parser.add_argument("--difficulty", type=str, required=True,
                    choices=["easy", "hard"])
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Velocity-HikingSafetyLite-G1-Play-v0")
parser.add_argument("--load_run", type=str, default="2026-06-07_00-16-00_g1_hiking_safety_lite_3000")
parser.add_argument("--checkpoint", type=str, default="model_2999.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from isaaclab.terrains import (
    TerrainGeneratorCfg,
    MeshPyramidStairsTerrainCfg,
    MeshGapTerrainCfg,
    HfSteppingStonesTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
)

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TERRAIN_CFGS = {
    "stairs": MeshPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.18),
        step_width=0.35,
        platform_width=2.0,
        border_width=0.5,
        holes=False,
    ),
    "gaps": MeshGapTerrainCfg(
        proportion=1.0,
        gap_width_range=(0.15, 0.45),
        platform_width=2.0,
    ),
    "stones": HfSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_height_max=0.08,
        stone_width_range=(0.35, 0.55),
        stone_distance_range=(0.05, 0.25),
        holes_depth=-0.8,
        platform_width=2.0,
    ),
    "boxes": MeshRepeatedBoxesTerrainCfg(
        proportion=1.0,
        object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=4, height=0.05, size=(0.3, 0.3),
        ),
        object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=10, height=0.20, size=(0.5, 0.5),
        ),
        abs_height_noise=(-0.02, 0.02),
        platform_width=2.0,
    ),
}

DIFFICULTY = {"easy": 2, "hard": 8}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    terrain_type = args_cli.terrain_type
    difficulty = args_cli.difficulty

    terrain_cfg = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=5,
        num_cols=5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={terrain_type: TERRAIN_CFGS[terrain_type]},
    )

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 30.0
    env_cfg.scene.terrain.terrain_generator = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = DIFFICULTY[difficulty]
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.6)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    ckpt_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Checkpoint: HikingSafetyLite model_2999.pt")
    print(f"[INFO] Terrain: {terrain_type}, Difficulty: {difficulty}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    out_name = os.path.splitext(os.path.basename(args_cli.out))[0]
    os.makedirs(out_dir, exist_ok=True)

    env = RecordVideo(
        env, video_folder=out_dir, name_prefix=out_name,
        step_trigger=lambda s: s == 0, video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    for step in range(max_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if dones.any():
            break

    env.close()
    simulation_app.close()
    print(f"[DONE] {args_cli.out}")


if __name__ == "__main__":
    main()
