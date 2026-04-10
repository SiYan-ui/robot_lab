# Project Guidelines

## Build and Run

- This repository is an Isaac Lab extension. Keep this repo outside the IsaacLab source tree.
- Install in editable mode with a Python environment that already has Isaac Lab and Isaac Sim dependencies:
  - `python -m pip install -e source/robot_lab`
- To configure VS Code Python paths for Isaac Sim, run task `setup_python_env` from `.vscode/tasks.json`.
- Useful runtime commands:
  - List environments: `python scripts/tools/list_envs.py`
  - Train (RSL-RL): `python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless`
  - Play (RSL-RL): `python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME>`
  - Experimental backends are under `scripts/reinforcement_learning/cusrl/` and `scripts/reinforcement_learning/skrl/`.
- Run code quality checks before proposing changes:
  - `pre-commit run --all-files`

## Architecture

- Core package code is under `source/robot_lab/robot_lab/`.
- Robot asset configs live in `source/robot_lab/robot_lab/assets/`.
- Task definitions live in `source/robot_lab/robot_lab/tasks/`:
  - `manager_based/` is the primary pattern for locomotion tasks.
  - `direct/` contains direct-task implementations (for example AMP-related tasks).
- Raw robot models (URDF/meshes) live in `source/robot_lab/data/Robots/`.
- Training and inference entry scripts live in `scripts/reinforcement_learning/`.

## Project-Specific Conventions

- Isaac Sim app launch order matters in training/play scripts:
  - Parse CLI args, call `AppLauncher.add_app_launcher_args`, create `AppLauncher`, then import/run the rest.
  - Do not "clean up" this ordering; `E402` is intentionally ignored in lint config to allow it.
- Environment registration pattern:
  - Register tasks via `gym.register(...)` inside task config package `__init__.py` files.
  - Use `kwargs` entry points for environment config and RL backend configs (for example `rsl_rl_cfg_entry_point`, `cusrl_cfg_entry_point`).
- Keep logs/checkpoints under existing backend-specific structure (for example `logs/rsl_rl/...`).
- Respect lint/type tooling configured in `pyproject.toml`:
  - Ruff line length is 120.
  - `__init__.py` may intentionally keep `F401` imports for registration side effects.

## References

- Main setup, usage, troubleshooting, and task-adding guide: `README.md`
- Lint/type/test tool configuration: `pyproject.toml`
- Pre-commit hooks: `.pre-commit-config.yaml`
- PR checklist expectations: `.github/PULL_REQUEST_TEMPLATE.md`
