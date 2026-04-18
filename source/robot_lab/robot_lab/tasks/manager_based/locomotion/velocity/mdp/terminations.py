# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_roll_pitch(
    env: ManagerBasedRLEnv,
    roll_threshold: float,
    pitch_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate immediately when roll/pitch exceeds independent thresholds."""
    asset = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
    roll = torch.atan2(projected_gravity_b[:, 1], -projected_gravity_b[:, 2])
    pitch = torch.atan2(projected_gravity_b[:, 0], -projected_gravity_b[:, 2])
    return torch.logical_or(torch.abs(roll) > roll_threshold, torch.abs(pitch) > pitch_threshold)


def illegal_contact_persistent(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    hold_time_s: float,
) -> torch.Tensor:
    """Terminate only when illegal contact lasts continuously for a long duration.

    Compared with immediate illegal-contact termination, this term allows short
    transients and terminates an environment only when sustained contact reaches
    ``hold_time_s`` seconds.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    over_threshold = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    persistent_contact_time = torch.where(over_threshold, current_contact_time, torch.zeros_like(current_contact_time))
    max_contact_time = torch.max(persistent_contact_time, dim=1)[0]
    return max_contact_time >= hold_time_s


def illegal_pose_persistent(
    env: ManagerBasedRLEnv,
    threshold: float,
    hold_time_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll/pitch exceeds ``threshold`` continuously for ``hold_time_s`` seconds."""
    asset = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
    roll = torch.atan2(projected_gravity_b[:, 1], -projected_gravity_b[:, 2])
    pitch = torch.atan2(projected_gravity_b[:, 0], -projected_gravity_b[:, 2])
    bad_pose = torch.logical_or(torch.abs(roll) > threshold, torch.abs(pitch) > threshold)

    if not hasattr(env, "_illegal_pose_duration"):
        env._illegal_pose_duration = torch.zeros(env.num_envs, device=env.device)

    env._illegal_pose_duration = torch.where(
        bad_pose,
        env._illegal_pose_duration + env.step_dt,
        torch.zeros_like(env._illegal_pose_duration),
    )
    return env._illegal_pose_duration >= hold_time_s
