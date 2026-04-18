# Go2 训练任务参数总览与 Isaac Lab 迁移指南

本文档面向两类需求：

1. 快速看懂本仓库 Go2 任务在训练时到底使用了哪些参数。
2. 把这套任务迁移到 Isaac Lab 的另一台机器人时，知道应该改哪里、怎么对齐。

## 1. 参数来源与生效路径

Go2 任务由以下文件共同决定：

- 任务注册与入口：`legged_gym/envs/__init__.py`（任务名 `go2`）
- Go2 专属覆盖：`legged_gym/envs/go2/go2_config.py`
- 通用默认参数：`legged_gym/envs/base/legged_robot_config.py`
- 真正执行逻辑（观测/奖励/终止/重置/控制）：`legged_gym/envs/base/legged_robot.py`
- 训练入口：`legged_gym/scripts/train.py`
- 命令行覆盖：`legged_gym/utils/helpers.py`

训练命令示例：

```bash
python legged_gym/scripts/train.py --task=go2
```

## 2. Go2 任务最终生效参数（按模块）

以下为“继承基类 + Go2 覆盖”后的有效参数。

### 2.1 环境与时间尺度

| 项 | 值 |
|---|---|
| `env.num_envs` | `4096` |
| `env.num_observations` | `48` |
| `env.num_privileged_obs` | `None` |
| `env.num_actions` | `12` |
| `env.episode_length_s` | `20` |
| `sim.dt` | `0.005 s` |
| `control.decimation` | `4` |
| 策略步长 `dt_policy` | `0.02 s` (`4 * 0.005`) |
| 策略频率 | `50 Hz` |
| 每回合步数 | `1000` (`ceil(20 / 0.02)`) |

### 2.2 初始状态（Go2 覆盖）

- 基座初始位置：`[0.0, 0.0, 0.42]`
- 初始姿态四元数：`[0.0, 0.0, 0.0, 1.0]`（沿用基类）
- 初始线速度/角速度：全 0（沿用基类）

默认关节角（action=0 时目标位）：

| 关节 | 角度(rad) |
|---|---|
| FL_hip_joint | 0.1 |
| RL_hip_joint | 0.1 |
| FR_hip_joint | -0.1 |
| RR_hip_joint | -0.1 |
| FL_thigh_joint | 0.8 |
| RL_thigh_joint | 1.0 |
| FR_thigh_joint | 0.8 |
| RR_thigh_joint | 1.0 |
| FL_calf_joint | -1.5 |
| RL_calf_joint | -1.5 |
| FR_calf_joint | -1.5 |
| RR_calf_joint | -1.5 |

### 2.3 控制与动作

| 项 | 值 |
|---|---|
| `control_type` | `P` |
| `stiffness` | `{'joint': 20.0}` |
| `damping` | `{'joint': 0.5}` |
| `action_scale` | `0.25` |
| `clip_actions` | `100.0` |

控制公式（每个 physics step 执行一次，且每个策略步执行 4 次）：

$$
\tau = K_p \cdot (a \cdot 0.25 + q_{default} - q) - K_d \cdot \dot{q}
$$

其中 `Kp=20`, `Kd=0.5`（Go2 所有 DOF 名都包含 `joint`，因此匹配同一组增益）。

扭矩最终按 URDF 力矩上限裁剪。

### 2.4 资产与接触规则（Go2 覆盖）

| 项 | 值 |
|---|---|
| 机器人文件 | `resources/robots/go2/urdf/go2.urdf` |
| `asset.name` | `go2` |
| `foot_name` | `foot` |
| 惩罚接触 body 关键词 | `['thigh', 'calf']` |
| 终止接触 body 关键词 | `['base']` |
| `self_collisions` | `1`（禁用自碰撞） |

### 2.5 指令采样（commands）

| 项 | 值 |
|---|---|
| `num_commands` | `4`（`lin_vel_x`, `lin_vel_y`, `ang_vel_yaw`, `heading`） |
| `resampling_time` | `10 s` |
| 重采样步数 | `500` (`10 / 0.02`) |
| `heading_command` | `True` |
| `lin_vel_x` 范围 | `[-1.0, 1.0]` m/s |
| `lin_vel_y` 范围 | `[-1.0, 1.0]` m/s |
| `ang_vel_yaw` 范围 | `[-1.0, 1.0]` rad/s |
| `heading` 范围 | `[-3.14, 3.14]` rad |

注意：`heading_command=True` 时，真实使用的 yaw 指令由 heading 误差在线计算：

$$
\omega_{yaw}^{cmd} = clip\left(0.5 \cdot wrap\_to\_pi(heading^{cmd} - heading^{robot}),\,-1,\,1\right)
$$

并且平面速度命令模长小于 0.2 时会被强制置零。

### 2.6 域随机化与扰动

| 项 | 值 |
|---|---|
| 摩擦随机化 | 开启 (`[0.5, 1.25]`) |
| 基座质量随机化 | 关闭 |
| 推搡机器人 | 开启 |
| `push_interval_s` | `15 s` |
| 推搡步数 | `750` (`15 / 0.02`) |
| `max_push_vel_xy` | `1.0` |

额外重置随机化（每次 reset）：

- 关节位置：`default_dof_pos * U(0.5, 1.5)`
- 根部线/角速度：`U(-0.5, 0.5)`

### 2.7 观测定义（48 维）

Go2 使用基类 `compute_observations`，观测拼接顺序如下：

$$
obs = [
v_{base}^{lin}(3),\
\omega_{base}(3),\
g_{proj}(3),\
cmd(3),\
q-q_{default}(12),\
\dot{q}(12),\
a_{last}(12)
]
$$

准确到实现的缩放：

| 片段 | 维度 | 缩放 |
|---|---|---|
| `base_lin_vel` | 3 | `obs_scales.lin_vel = 2.0` |
| `base_ang_vel` | 3 | `obs_scales.ang_vel = 0.25` |
| `projected_gravity` | 3 | 1.0 |
| `commands[:, :3]` | 3 | `[2.0, 2.0, 0.25]` |
| `dof_pos - default_dof_pos` | 12 | `obs_scales.dof_pos = 1.0` |
| `dof_vel` | 12 | `obs_scales.dof_vel = 0.05` |
| `actions` | 12 | 1.0 |

观测噪声（均匀噪声，按维度乘以 `noise_scale_vec`）：

| 片段 | 噪声幅值 |
|---|---|
| `base_lin_vel` | `0.1 * 1.0 * 2.0 = 0.2` |
| `base_ang_vel` | `0.2 * 1.0 * 0.25 = 0.05` |
| `projected_gravity` | `0.05` |
| `commands` | `0` |
| `dof_pos` | `0.01` |
| `dof_vel` | `1.5 * 0.05 = 0.075` |
| `actions` | `0` |

### 2.8 奖励/惩罚项（Go2 实际生效）

关键实现细节：

1. 奖励系数在运行时会乘以 `dt_policy=0.02`。
2. 系数等于 `0` 的项会被直接删除，不参与计算。
3. `only_positive_rewards=True`：总奖励先做 `clip(min=0)`，再叠加 `termination`（如果该项存在）。

Go2 的非零项如下（含 dt 后权重）：

| 奖励项 | 原始公式（每步） | 配置系数 | 生效系数(乘 dt) | 类型 |
|---|---|---:|---:|---|
| `tracking_lin_vel` | $\exp(-\|cmd_{xy}-v_{xy}\|^2/\sigma)$ | 1.0 | 0.02 | 奖励 |
| `tracking_ang_vel` | $\exp(-(cmd_{yaw}-\omega_z)^2/\sigma)$ | 0.5 | 0.01 | 奖励 |
| `lin_vel_z` | $v_z^2$ | -2.0 | -0.04 | 惩罚 |
| `ang_vel_xy` | $\omega_x^2+\omega_y^2$ | -0.05 | -0.001 | 惩罚 |
| `torques` | $\sum \tau^2$ | -0.0002 | -0.000004 | 惩罚 |
| `dof_acc` | $\sum ((\dot q_{t-1}-\dot q_t)/dt)^2$ | -2.5e-7 | -5e-9 | 惩罚 |
| `feet_air_time` | 首次触地时累积腾空时长奖励 | 1.0 | 0.02 | 奖励 |
| `collision` | 惩罚 body 接触计数(thigh/calf) | -1.0 | -0.02 | 惩罚 |
| `action_rate` | $\sum (a_{t-1}-a_t)^2$ | -0.01 | -0.0002 | 惩罚 |
| `dof_pos_limits` | 超软限位距离和 | -10.0 | -0.2 | 强惩罚 |

其中 `tracking_sigma = 0.25`。

Go2 中配置为 0（因此删除）的项：

- `termination`
- `orientation`
- `dof_vel`
- `base_height`
- `feet_stumble`
- `stand_still`

补充注意：

- 代码里奖励函数名是 `_reward_stumble`，而配置键是 `feet_stumble`。当前因为系数是 0 不触发；若将 `feet_stumble` 设为非 0，会因为找不到 `_reward_feet_stumble` 而报错。

### 2.9 终止条件

满足任一即重置：

1. 任一 `terminate_after_contacts_on` body 的接触力范数 > 1。
2. 姿态阈值越界：`abs(pitch) > 1.0` 或 `abs(roll) > 0.8`。
3. 超时：`episode_length > max_episode_length`。

### 2.10 训练器（PPO）参数

Go2 在 `GO2RoughCfgPPO` 仅覆盖了 `experiment_name='rough_go2'`（算法熵系数与基类同为 `0.01`）。

核心训练超参：

| 模块 | 参数 | 值 |
|---|---|---|
| policy | `actor_hidden_dims` | `[512, 256, 128]` |
| policy | `critic_hidden_dims` | `[512, 256, 128]` |
| policy | `activation` | `elu` |
| algorithm | `learning_rate` | `1e-3` |
| algorithm | `gamma` / `lam` | `0.99` / `0.95` |
| algorithm | `num_learning_epochs` | `5` |
| algorithm | `num_mini_batches` | `4` |
| algorithm | `clip_param` | `0.2` |
| algorithm | `entropy_coef` | `0.01` |
| runner | `num_steps_per_env` | `24` |
| runner | `max_iterations` | `1500` |
| runner | `experiment_name` | `rough_go2` |

按默认 `num_envs=4096` 估算：

- 每次迭代采样步数：`4096 * 24 = 98304`
- 每次迭代时间窗：`24 * 0.02 = 0.48 s`

### 2.11 命令行可覆盖参数

通过 `helpers.update_cfg_from_args` 可覆盖：

- `env.num_envs`
- `train.seed`
- `runner.max_iterations`
- `runner.resume/experiment_name/run_name/load_run/checkpoint`

### 2.12 其余关键默认参数（Go2 未覆盖但实际沿用）

#### 地形与场景

| 项 | 值 | 备注 |
|---|---|---|
| `terrain.mesh_type` | `plane` | 当前 `create_sim` 走平地创建流程 |
| `terrain.static_friction` | `1.0` | 生效 |
| `terrain.dynamic_friction` | `1.0` | 生效 |
| `terrain.restitution` | `0.0` | 生效 |
| `terrain.measure_heights` | `True` | 当前 Go2 观测未使用高度采样 |
| `terrain.horizontal_scale` | `0.1` | 仅在高度场/三角网格场景有意义 |
| `terrain.vertical_scale` | `0.005` | 仅在高度场/三角网格场景有意义 |
| `terrain.curriculum` | `True` | 当前代码路径未实际更新地形课程 |

#### 奖励全局超参数

| 项 | 值 |
|---|---|
| `rewards.only_positive_rewards` | `True` |
| `rewards.tracking_sigma` | `0.25` |
| `rewards.soft_dof_pos_limit` | `0.9`（Go2 覆盖） |
| `rewards.soft_dof_vel_limit` | `1.0` |
| `rewards.soft_torque_limit` | `1.0` |
| `rewards.base_height_target` | `0.25`（Go2 覆盖） |
| `rewards.max_contact_force` | `100.0` |

#### 归一化与噪声

| 项 | 值 |
|---|---|
| `normalization.obs_scales.lin_vel` | `2.0` |
| `normalization.obs_scales.ang_vel` | `0.25` |
| `normalization.obs_scales.dof_pos` | `1.0` |
| `normalization.obs_scales.dof_vel` | `0.05` |
| `normalization.obs_scales.height_measurements` | `5.0` |
| `normalization.clip_observations` | `100.0` |
| `normalization.clip_actions` | `100.0` |
| `noise.add_noise` | `True` |
| `noise.noise_level` | `1.0` |
| `noise.noise_scales.dof_pos` | `0.01` |
| `noise.noise_scales.dof_vel` | `1.5` |
| `noise.noise_scales.lin_vel` | `0.1` |
| `noise.noise_scales.ang_vel` | `0.2` |
| `noise.noise_scales.gravity` | `0.05` |
| `noise.noise_scales.height_measurements` | `0.1` |

#### 仿真与 PhysX

| 项 | 值 |
|---|---|
| `sim.substeps` | `1` |
| `sim.gravity` | `[0, 0, -9.81]` |
| `physx.solver_type` | `1` (TGS) |
| `physx.num_position_iterations` | `4` |
| `physx.num_velocity_iterations` | `0` |
| `physx.contact_offset` | `0.01` |
| `physx.rest_offset` | `0.0` |
| `physx.max_depenetration_velocity` | `1.0` |

#### 指令课程参数（当前默认不启用）

| 项 | 值 | 备注 |
|---|---|---|
| `commands.curriculum` | `False` | 默认关闭 |
| `commands.max_curriculum` | `1.0` | 仅课程启用时使用 |

注：基类里虽然有 `update_command_curriculum`，但当前重置流程中未调用该函数，因此 Go2 默认不会自动扩展命令范围。

## 3. 与 Isaac Lab 迁移对照（迁移到另一台机器人）

下面给出“对齐维度”的迁移模板。迁移核心不是逐行搬代码，而是对齐行为学定义。

### 3.1 一页式迁移清单

1. 机器人定义：替换 USD/URDF 与关节名，先冻结奖励，只验证可稳定站立。
2. 动作空间：确认动作维度、关节顺序、`default_joint_angles`、`action_scale` 与控制频率。
3. 观测空间：严格按顺序迁移 48 维（或新维度），同步噪声向量与归一化。
4. 指令系统：保留 `heading_command` 逻辑和重采样周期，不要直接拿 `ang_vel_yaw` 范围替代。
5. 奖励项：先迁移 Go2 的 10 个非零项，再根据新机器人逐项开关和调权重。
6. 终止条件：接触终止 body + 姿态阈值 + 超时三者都要保留。
7. 域随机化：至少保留摩擦随机化与 push 事件，再逐步加质量随机化。
8. PPO 超参：先复用 Go2 默认值，待行为稳定后再改网络规模和 batch。

### 3.2 参数映射关系（Legged Gym -> Isaac Lab 概念）

| Legged Gym 概念 | Isaac Lab 迁移对象（概念层） | 迁移要点 |
|---|---|---|
| `init_state.default_joint_angles` | 机器人初始关节状态配置 | 名称需 1:1 对齐新机器人关节名 |
| `control` + `_compute_torques` | ActionTerm + Actuator/控制器 | 要保持同样的控制频率和缩放 |
| `compute_observations` | ObservationTerm 列表 | 顺序一变，策略就不兼容 |
| `_get_noise_scale_vec` | 观测噪声配置 | 维度改变时必须重写噪声向量 |
| `commands` + `_resample_commands` | CommandManager | 保留 heading 到 yaw 的在线转换 |
| `_reward_*` + `rewards.scales` | RewardTerm 列表 | 注意本仓库有“系数乘 dt”逻辑 |
| `check_termination` | TerminationTerm 列表 | 接触终止与姿态阈值都要迁移 |
| `domain_rand` + reset 随机化 | Randomization / EventTerm | 训练鲁棒性的关键来源 |

### 3.3 建议的迁移顺序（实操）

1. 先做“行为等价迁移”：只替换机器人模型和关节名，其余逻辑尽量不动。
2. 做 3 组冒烟测试：
   - 静止命令（0,0,0）是否能稳定站立 20s。
   - 匀速命令（如 0.5, 0, 0）是否持续跟踪。
   - 随机命令时是否出现早期大规模跌倒。
3. 观察每个奖励分量占比，确保 `tracking_*` 与惩罚项数量级接近 Go2。
4. 再进行机器人特化：
   - 调整 `base_height_target`。
   - 调整 `soft_dof_pos_limit` 与关节限位惩罚。
   - 视硬件能力增减 torque/action_rate 惩罚。

### 3.4 最容易踩坑的 8 点

1. 忘记奖励系数乘 `dt`，导致奖励量级偏离。
2. 忽略 `only_positive_rewards`，使训练初期被惩罚主导。
3. 改了观测维度但没改噪声向量切片。
4. 忽略 `heading_command=True` 的 yaw 重写逻辑。
5. 动作维度对上了但关节顺序错位。
6. reset 时漏掉关节/根速度随机化，策略泛化显著下降。
7. 终止 body 名称未匹配新机器人，导致该终止规则失效。
8. 把配置里的“地形参数”当作已生效功能（当前 Go2 代码路径主要是平地 `create_ground_plane`）。

## 4. 迁移后验收标准（建议）

建议至少满足以下指标再进入大规模训练：

1. 维度一致：`obs_dim`、`act_dim`、命令维度全部与实现一致。
2. 统计稳定：奖励总和、各子项均值、终止率在可控区间。
3. 行为可解释：静止、前进、转向三类命令都能表现正确。
4. 鲁棒性有效：开启摩擦随机化与 push 后性能下降可接受。

---

如果你准备实际迁移到 Isaac Lab 的某个具体机器人（例如另一款四足或人形），建议先按本文件第 3.1 清单做“最小可运行版本”，再进行奖励权重细调。