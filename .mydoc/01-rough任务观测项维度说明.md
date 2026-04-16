# Rough任务观测项维度说明（RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0）

## 结论速览
- 当前你使用的 rough 任务：`RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0`
- 该任务的 `policy` 观测总维度：**45**
- 该任务的 `critic` 观测总维度：**235**

## 1. 基础观测模板（来自速度任务基类）
基类 `LocomotionVelocityRoughEnvCfg` 的观测项声明顺序（`concatenate_terms=True`，按声明顺序拼接）为：
1. `base_lin_vel`
2. `base_ang_vel`
3. `projected_gravity`
4. `velocity_commands`
5. `joint_pos`
6. `joint_vel`
7. `actions`
8. `height_scan`

对应维度（D1任务）为：
- `base_lin_vel`: 3
- `base_ang_vel`: 3
- `projected_gravity`: 3
- `velocity_commands`: 3（来自 `base_velocity` 命令：`lin_vel_x, lin_vel_y, ang_vel_z`）
- `joint_pos`: 12（D1配置里 `joint_names` 共12个关节）
- `joint_vel`: 12（同上）
- `actions`: 12（动作与受控关节一一对应）
- `height_scan`: 187（`GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`，17×11）

完整模板总维度：
- `3 + 3 + 3 + 3 + 12 + 12 + 12 + 187 = 235`

## 2. D1 rough任务对观测的覆写
在 `AgibotD1RoughEnvCfg` 中：
- `self.observations.policy.base_lin_vel = None`
- `self.observations.policy.height_scan = None`

即：**仅 policy 组删除了这两项**；critic 组仍保留完整模板。

## 3. 最终拼接顺序与维度
### 3.1 Policy观测（Actor输入）
按顺序拼接：
1. `base_ang_vel` (3)
2. `projected_gravity` (3)
3. `velocity_commands` (3)
4. `joint_pos` (12)
5. `joint_vel` (12)
6. `actions` (12)

总维度：
- `3 + 3 + 3 + 12 + 12 + 12 = 45`

### 3.2 Critic观测（Critic输入）
按顺序拼接：
1. `base_lin_vel` (3)
2. `base_ang_vel` (3)
3. `projected_gravity` (3)
4. `velocity_commands` (3)
5. `joint_pos` (12)
6. `joint_vel` (12)
7. `actions` (12)
8. `height_scan` (187)

总维度：
- `3 + 3 + 3 + 3 + 12 + 12 + 12 + 187 = 235`

## 4. 代码依据
- 观测模板与拼接机制：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- D1 rough 对 policy 观测的删项：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/agibot_d1/rough_env_cfg.py`
- D1任务注册ID：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/agibot_d1/__init__.py`
