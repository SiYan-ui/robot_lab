# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.etc import XSDOG_CFG  # isort: skip


@configclass
class XSDogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "rf_hip", "rf_thigh", "rf_calf",
        "lf_hip", "lf_thigh", "lf_calf",
        "rh_hip", "rh_thigh", "rh_calf",
        "lh_hip", "lh_thigh", "lh_calf",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = XSDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 1.0
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip": 0.125, "^(?!.*_hip).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.05, 0.15),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-3.0, 3.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -50.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.4
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -5e-6
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -5e-6
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["rf_(hip|thigh|calf).*", "lh_(hip|thigh|calf).*"],
            ["lf_(hip|thigh|calf).*", "rh_(hip|thigh|calf).*"],
        ]

        self.rewards.action_rate_l2.weight = -0.03

        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.illegal_contact_duration_penalty.weight = -20.0
        self.rewards.illegal_contact_duration_penalty.params["sensor_cfg"].body_names = [self.base_link_name]
        self.rewards.illegal_contact_duration_penalty.params["threshold"] = 1.0
        self.rewards.illegal_contact_duration_penalty.params["hold_time_s"] = 10.0
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 1.75

        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("LF_foot", "RH_foot"), ("RF_foot", "LH_foot"))
        self.rewards.upward.weight = 1.0
        self.rewards.excessive_roll_penalty.weight = -10.0

        # Keep training command ranges normalized to [-1, 1].
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        if self.__class__.__name__ == "XSDogRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = self.base_link_name
        self.terminations.illegal_contact.func = mdp.illegal_contact_persistent
        self.terminations.illegal_contact.params["threshold"] = 1.0
        self.terminations.illegal_contact.params["hold_time_s"] = 5.0
        self.terminations.illegal_pose = DoneTerm(
            func=mdp.illegal_pose_persistent,
            params={
                "threshold": 1.0,
                "hold_time_s": 8.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel = None
        # self.curriculum.command_levels_ang_vel = None


@configclass
class XSDogV1RoughEnvCfg(XSDogRoughEnvCfg):
    """XSDog rough task aligned with the Xiaotian-v1 reward and termination layout."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # Keep XSDog nominal standing pose while using v1 task layout.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.50)
        self.scene.robot.init_state.joint_pos = {
            ".*_hip": 0.0,
            ".*_thigh": 0.8,
            ".*_calf": -1.5,
        }

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel = ObsTerm(
            func=env_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.velocity_commands.func = mdp.scaled_generated_commands
        self.observations.policy.velocity_commands.params = {
            "command_name": "base_velocity",
            "lin_vel_scale": 2.0,
            "ang_vel_scale": 0.25,
        }
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.velocity_commands.func = mdp.scaled_generated_commands
        self.observations.critic.velocity_commands.params = {
            "command_name": "base_velocity",
            "lin_vel_scale": 2.0,
            "ang_vel_scale": 0.25,
        }
        self.observations.critic.height_scan = None
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_joints.params = {
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_push_robot.interval_range_s = (15.0, 15.0)
        self.events.randomize_push_robot.params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = 0

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.4
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 23Nm -> 100Nm scaling for torque-related penalty keeps utilization pressure comparable.
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = 0
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.joint_mirror.weight = 0
        self.rewards.action_mirror.weight = 0
        self.rewards.action_sync.weight = 0

        self.rewards.applied_torque_limits.weight = 0
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh.*", ".*calf.*"]
        self.rewards.illegal_contact_duration_penalty.weight = 0
        self.rewards.contact_forces.weight = 0

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_distance_y_exp.weight = 0
        self.rewards.feet_gait.weight = 0
        self.rewards.upward.weight = 0
        self.rewards.excessive_roll_penalty.weight = 0

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ------------------------------Terminations------------------------------
        self.terminations.terrain_out_of_bounds = None
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_pose = DoneTerm(
            func=mdp.illegal_roll_pitch,
            params={
                "roll_threshold": 0.8,
                "pitch_threshold": 1.0,
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "XSDogV1RoughEnvCfg":
            self.disable_zero_weight_rewards()
