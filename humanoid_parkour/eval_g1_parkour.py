#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stratified evaluation script for G1 parkour locomotion agent.

Evaluates models on Easy/Medium/Hard terrain levels with comprehensive metrics:
  success rate, fall rate, velocity error, distance traveled, time survived,
  terrain level reached, episode return, track reward.

Usage:
    ./isaaclab.sh -p tools/eval_g1_parkour.py \
        --task Isaac-Velocity-Parkour-G1-Play-v0 \
        --terrain_level easy --num_trials 50 \
        --load_run <run> --checkpoint model_3000.pt --out eval_easy.csv
"""

import argparse
import sys

from isaaclab.app import AppLauncher

TERRAIN_LEVELS = {"easy": 2, "medium": 5, "hard": 8, "full": None}

parser = argparse.ArgumentParser(description="Stratified evaluation of G1 parkour agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Parkour-G1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_2999.pt")
parser.add_argument("--num_trials", type=int, default=50)
parser.add_argument("--out", type=str, default="eval_results.csv")
parser.add_argument("--terrain_level", type=str, default="full",
                    choices=["easy", "medium", "hard", "full"],
                    help="Terrain difficulty: easy(0-2) medium(0-5) hard(0-8) full(all)")
parser.add_argument("--eval_level", type=int, default=None,
                    help="Fixed single terrain level (0-9). Overrides --terrain_level.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import os
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab.managers import SceneEntityCfg

from humanoid_parkour.hiking_mdp import _foothold_safe_mask, _detect_first_contact, _nearest_height_patch


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    terrain_label = args_cli.terrain_level
    max_tl = TERRAIN_LEVELS[terrain_label]

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    if args_cli.eval_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.eval_level
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        terrain_label = f'level_{args_cli.eval_level}'
        max_tl = args_cli.eval_level
    else:
        env_cfg.scene.terrain.max_init_terrain_level = max_tl
        if max_tl is None:
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Terrain level: {terrain_label} (max_init_terrain_level={max_tl})")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", f"eval_{terrain_label}"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    n_envs = args_cli.num_envs
    device = env.unwrapped.device

    episode_dist = torch.zeros(n_envs, device=device)
    episode_steps = torch.zeros(n_envs, device=device)
    episode_terrain = torch.zeros(n_envs, device=device)
    episode_rew_sum = torch.zeros(n_envs, device=device)
    fall_positions = torch.zeros(n_envs, 3, device=device)

    episode_safe_td = torch.zeros(n_envs, device=device)
    episode_unsafe_td = torch.zeros(n_envs, device=device)
    episode_swing_violation_steps = torch.zeros(n_envs, device=device)
    episode_swing_steps = torch.zeros(n_envs, device=device)

    height_cfg = SceneEntityCfg("height_scanner")
    contact_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    asset_cfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")

    results = []
    done_count = 0
    step_count = 0

    print(f"\n[INFO] Eval: {terrain_label}, {args_cli.num_trials} episodes, max {max_steps} steps/ep\n")

    while simulation_app.is_running() and done_count < args_cli.num_trials:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        log_dict = extras.get("log", {})

        base_vel_w = env.unwrapped.scene["robot"].data.root_lin_vel_w[:, :2]
        episode_dist += torch.norm(base_vel_w * dt, dim=-1)
        episode_steps += 1
        episode_rew_sum += rew

        safe_mask, _ = _foothold_safe_mask(
            env.unwrapped, height_cfg, asset_cfg, k=9, foot_radius=0.12,
            max_height_var=0.055, max_foot_terrain_gap=0.16, min_support_rays=3,
        )
        first_contact = _detect_first_contact(env.unwrapped, contact_cfg)
        episode_safe_td += (first_contact & safe_mask).float().sum(dim=1)
        episode_unsafe_td += (first_contact & (~safe_mask)).float().sum(dim=1)

        patch_z, _, foot_pos_w = _nearest_height_patch(env.unwrapped, height_cfg, asset_cfg, k=1)
        terrain_z = patch_z[..., 0]
        clearance = foot_pos_w[..., 2] - terrain_z
        air_time = env.unwrapped.scene.sensors["contact_forces"].data.current_air_time[:, contact_cfg.body_ids]
        in_swing = air_time > 0.03
        clearance_violation = (clearance < 0.08) & in_swing
        episode_swing_violation_steps += clearance_violation.float().sum(dim=1)
        episode_swing_steps += in_swing.float().sum(dim=1)

        tl = log_dict.get("Curriculum/terrain_levels")
        if tl is not None:
            tl_val = float(tl) if not isinstance(tl, torch.Tensor) else tl
            episode_terrain = torch.maximum(episode_terrain, torch.tensor(tl_val, device=device))

        dones_cpu = dones.cpu()
        if dones_cpu.any():
            done_indices = dones_cpu.nonzero(as_tuple=False).squeeze(-1)
            for env_id in done_indices.tolist():
                if done_count >= args_cli.num_trials:
                    break

                ep_steps = float(episode_steps[env_id].item())
                fell = ep_steps < max_steps * 0.9

                vel_err_xy = float(log_dict.get("Metrics/base_velocity/error_vel_xy", -1))
                vel_err_yaw = float(log_dict.get("Metrics/base_velocity/error_vel_yaw", -1))
                track_rw = float(log_dict.get("Episode_Reward/track_lin_vel_xy_exp", 0.0))
                ep_tl_val = float(log_dict.get("Curriculum/terrain_levels", episode_terrain[env_id].item()))

                try:
                    root_pos = env.unwrapped.scene["robot"].data.root_pos_w[env_id]
                    fall_positions[env_id] = root_pos.clone()
                except Exception:
                    pass

                survived_time = float(episode_steps[env_id].item()) * dt
                dist_traveled = float(episode_dist[env_id].item())

                safe_td = float(episode_safe_td[env_id].item())
                unsafe_td = float(episode_unsafe_td[env_id].item())
                total_td = safe_td + unsafe_td
                safe_td_rate = safe_td / total_td if total_td > 0 else -1.0

                total_swing = float(episode_swing_steps[env_id].item())
                swing_viol = float(episode_swing_violation_steps[env_id].item())
                swing_viol_rate = swing_viol / total_swing if total_swing > 0 else 0.0

                results.append({
                    "episode_idx": done_count,
                    "env_id": env_id,
                    "terrain": terrain_label,
                    "success": 0 if fell else 1,
                    "fall": 1 if fell else 0,
                    "velocity_error_xy": vel_err_xy,
                    "velocity_error_yaw": vel_err_yaw,
                    "distance_traveled": dist_traveled,
                    "time_survived": survived_time,
                    "terrain_level": ep_tl_val,
                    "episode_return": float(episode_rew_sum[env_id].item()),
                    "track_reward": track_rw,
                    "fall_pos_x": float(fall_positions[env_id, 0].item()),
                    "fall_pos_y": float(fall_positions[env_id, 1].item()),
                    "safe_touchdown_count": safe_td,
                    "unsafe_touchdown_count": unsafe_td,
                    "safe_touchdown_rate": safe_td_rate,
                    "swing_clearance_violation_steps": swing_viol,
                    "swing_clearance_violation_rate": swing_viol_rate,
                })
                done_count += 1

            episode_dist[done_indices] = 0.0
            episode_steps[done_indices] = 0
            episode_terrain[done_indices] = 0.0
            episode_rew_sum[done_indices] = 0.0
            episode_safe_td[done_indices] = 0.0
            episode_unsafe_td[done_indices] = 0.0
            episode_swing_violation_steps[done_indices] = 0.0
            episode_swing_steps[done_indices] = 0.0

        step_count += 1
        if step_count % 200 == 0:
            print(f"  Step {step_count}, done: {done_count}/{args_cli.num_trials}")

    env.close()
    print(f"\n  Total steps: {step_count}, episodes: {done_count}")

    total = len(results)
    if total == 0:
        print("[WARNING] No episodes completed.")
        return

    falls = sum(1 for r in results if r["fall"])
    success_rate = (total - falls) / total * 100

    vel_errors = [r["velocity_error_xy"] for r in results if r["velocity_error_xy"] >= 0]
    mean_vel_err = sum(vel_errors) / len(vel_errors) if vel_errors else -1

    dists = [r["distance_traveled"] for r in results]
    mean_dist = sum(dists) / len(dists) if dists else 0

    times = [r["time_survived"] for r in results]
    mean_time = sum(times) / len(times) if times else 0

    tl_vals = [r["terrain_level"] for r in results]
    mean_tl = sum(tl_vals) / len(tl_vals) if tl_vals else 0

    rets = [r["episode_return"] for r in results]
    mean_ret = sum(rets) / len(rets) if rets else 0

    trs = [r["track_reward"] for r in results]
    mean_tr = sum(trs) / len(trs) if trs else 0

    safe_td_rates = [r["safe_touchdown_rate"] for r in results if r["safe_touchdown_rate"] >= 0]
    mean_safe_td_rate = sum(safe_td_rates) / len(safe_td_rates) if safe_td_rates else -1

    total_safe_td = sum(r["safe_touchdown_count"] for r in results)
    total_unsafe_td = sum(r["unsafe_touchdown_count"] for r in results)

    viol_rates = [r["swing_clearance_violation_rate"] for r in results]
    mean_viol_rate = sum(viol_rates) / len(viol_rates) if viol_rates else 0

    print("\n" + "=" * 70)
    print(f"EVALUATION: {terrain_label.upper()} TERRAIN")
    print("=" * 70)
    print(f"  Mean track reward:  {mean_tr:.4f}")
    print(f"  Safe touchdown rate:{mean_safe_td_rate:.3f}" if mean_safe_td_rate >= 0 else f"  Safe touchdown rate: N/A (no touchdowns)")
    print(f"  Total safe/unsafe TD:{total_safe_td:.0f} / {total_unsafe_td:.0f}")
    print(f"  Mean swing viol rate:{mean_viol_rate:.4f}")
    print("=" * 70)

    csv_path = os.path.join(log_dir, args_cli.out)
    fieldnames = [
        "episode_idx", "env_id", "terrain", "success", "fall",
        "velocity_error_xy", "velocity_error_yaw", "distance_traveled",
        "time_survived", "terrain_level", "episode_return", "track_reward",
        "fall_pos_x", "fall_pos_y",
        "safe_touchdown_count", "unsafe_touchdown_count",
        "safe_touchdown_rate", "swing_clearance_violation_steps",
        "swing_clearance_violation_rate",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[INFO] Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
