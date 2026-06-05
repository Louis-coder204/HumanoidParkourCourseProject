from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def _nearest_height_patch(
    env,
    height_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    k: int = 9,
):
    """Return k nearest terrain ray heights around each foot.

    Returns:
        patch_z:    (num_envs, num_feet, k)
        patch_d2:   (num_envs, num_feet, k)
        foot_pos_w: (num_envs, num_feet, 3)
    """
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    robot = env.scene[asset_cfg.name]

    ray_hits_w = height_sensor.data.ray_hits_w
    ray_xy = ray_hits_w[..., :2]
    ray_z = ray_hits_w[..., 2]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_xy = foot_pos_w[..., :2]

    d2 = torch.sum((ray_xy[:, None, :, :] - foot_xy[:, :, None, :]) ** 2, dim=-1)

    valid = torch.isfinite(ray_z)
    d2 = torch.where(valid[:, None, :], d2, torch.full_like(d2, float("inf")))

    idx = torch.topk(d2, k=k, dim=-1, largest=False).indices

    ray_z_expand = ray_z[:, None, :].expand(-1, foot_xy.shape[1], -1)
    patch_z = torch.gather(ray_z_expand, dim=2, index=idx)
    patch_d2 = torch.gather(d2, dim=2, index=idx)

    return patch_z, patch_d2, foot_pos_w


def _foothold_safe_mask(
    env,
    height_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    k: int,
    foot_radius: float,
    max_height_var: float,
    max_foot_terrain_gap: float,
    min_support_rays: int,
):
    """Per-foot boolean safety mask, shape (N, F)."""
    patch_z, patch_d2, foot_pos_w = _nearest_height_patch(env, height_sensor_cfg, asset_cfg, k=k)

    center_z = patch_z[..., 0]
    z_max = torch.max(patch_z, dim=-1).values
    z_min = torch.min(patch_z, dim=-1).values
    height_var = z_max - z_min

    support = (
        (patch_d2 < foot_radius ** 2)
        & (torch.abs(patch_z - center_z.unsqueeze(-1)) < max_height_var)
        & torch.isfinite(patch_z)
    )
    support_count = torch.sum(support, dim=-1)

    foot_terrain_gap = torch.abs(foot_pos_w[..., 2] - center_z)

    safe = (
        (height_var < max_height_var)
        & (support_count >= min_support_rays)
        & (foot_terrain_gap < max_foot_terrain_gap)
    )
    return safe, center_z


def _detect_first_contact(env, sensor_cfg: SceneEntityCfg):
    """Detect when feet transition from air to ground.

    Returns boolean tensor of shape (N, F) where True means the foot
    just made first ground contact this step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    first_contact = (contact_time > 0.0) & (contact_time <= env.step_dt * 1.5)
    return first_contact


def _command_gate(env, command_name: str) -> torch.Tensor:
    """Boolean mask: True when velocity command magnitude exceeds threshold."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) > 0.1


def safe_touchdown(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    k: int = 9,
    foot_radius: float = 0.12,
    max_height_var: float = 0.055,
    max_foot_terrain_gap: float = 0.16,
    min_support_rays: int = 3,
):
    """Reward safe first contact.

    Reward is only emitted on touchdown events, never during continuous stance.
    """
    safe, _ = _foothold_safe_mask(
        env, height_sensor_cfg, asset_cfg, k, foot_radius,
        max_height_var, max_foot_terrain_gap, min_support_rays,
    )
    first_contact = _detect_first_contact(env, sensor_cfg)
    reward = torch.sum(first_contact.float() * safe.float(), dim=1)
    reward *= _command_gate(env, command_name)
    return reward


def unsafe_touchdown(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    k: int = 9,
    foot_radius: float = 0.12,
    max_height_var: float = 0.055,
    max_foot_terrain_gap: float = 0.16,
    min_support_rays: int = 3,
):
    """Penalty for landing on edge/gap/rough patch."""
    safe, _ = _foothold_safe_mask(
        env, height_sensor_cfg, asset_cfg, k, foot_radius,
        max_height_var, max_foot_terrain_gap, min_support_rays,
    )
    first_contact = _detect_first_contact(env, sensor_cfg)
    bad = first_contact & (~safe)
    penalty = torch.sum(bad.float(), dim=1)
    penalty *= _command_gate(env, command_name)
    return penalty


def swing_clearance(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    min_clearance: float = 0.08,
    k: int = 1,
):
    """Penalty when swing foot is too close to terrain height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    patch_z, _, foot_pos_w = _nearest_height_patch(env, height_sensor_cfg, asset_cfg, k=k)
    terrain_z = patch_z[..., 0]
    clearance = foot_pos_w[..., 2] - terrain_z

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_swing = air_time > 0.03

    penalty = torch.clamp(min_clearance - clearance, min=0.0)
    return torch.sum(penalty * in_swing.float(), dim=1)


def stance_edge_risk(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    k: int = 9,
    foot_radius: float = 0.12,
    max_height_var: float = 0.055,
    max_foot_terrain_gap: float = 0.16,
    min_support_rays: int = 3,
):
    """Small continuous penalty if stance foot remains on unsafe patch."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    safe, _ = _foothold_safe_mask(
        env, height_sensor_cfg, asset_cfg, k, foot_radius,
        max_height_var, max_foot_terrain_gap, min_support_rays,
    )

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    risk = contacts & (~safe)
    return torch.sum(risk.float(), dim=1)
