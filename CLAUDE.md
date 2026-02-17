# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bimanual SO101 robot teleoperation, data collection, and policy training/inference system built on LeRobot.

## Commands

```bash
uv sync                                    # Install dependencies
uv run ruff check .                        # Lint
uv run ruff check --fix .                  # Lint with auto-fix
uv run ruff format .                       # Format
uv run ty check                            # Type check

uv run data_taking/teleop.py --display     # Teleoperation with Rerun viz
uv run data_taking/record.py               # Record episodes
uv run training/train.py                   # Train ACT policy
uv run inference/run.py                    # Run policy on robot

uv run utils/find_cameras.py               # Discover cameras & assign stable paths
uv run utils/test_urdf_viz.py              # Test URDF visualization
uv run utils/test_realsense.py             # Test RealSense camera
```

## Architecture

### Module layout
- **lib/** — Shared core: config loading (`config.py`), robot factory functions (`robots.py`), Rerun URDF visualization (`urdf_viz.py`)
- **data_taking/** — Teleoperation, recording, replay scripts
- **training/** — ACT policy training with wandb tracking
- **inference/** — Policy execution on robot hardware
- **setup/** — One-time hardware setup: motor config, calibration, health monitoring
- **utils/** — Test/diagnostic scripts
- **lerobot/** — Git submodule, installed as editable (`uv` source)
- **SO-ARM100/** — Git submodule with URDF and 3D mesh files

### Configuration system
Three-level hierarchy: global `config.toml` (machine-specific, gitignored) for hardware ports/cameras/URDF, per-module `{module}/config.toml` for task-specific settings, code defaults as fallback. All TOML, loaded via `lib/config.py`.

### Key patterns
- **Bimanual naming**: Observations/actions use `{left,right}_{joint}.pos` keys (e.g. `left_shoulder_pan.pos`). Factory functions in `lib/robots.py` handle left/right arm wiring.
- **Calibration stored in repo**: `calibration/{leader,follower}/{arm}.json` — not gitignored, shared across machines.
- **Local dataset storage**: Datasets live in `data/datasets/{repo_id}` (not HF cache), enabling resume of incomplete recordings. `data/` is gitignored.
- **Git hash in datasets**: `lib/config.py:get_git_info()` embeds commit hash in dataset metadata for reproducibility.
- **URDF viz**: `lib/urdf_viz.py` parses URDF XML, builds Rerun entity hierarchies with proper kinematic chains. Uses a global `_global_visualizer` singleton. `init_rerun_with_urdf()` sets up the session, `log_observation_and_action()` updates it per frame.
- **Scripts use `sys.path.insert`** to import from `lib/` — not a package install.

## Code Style

- Python 3.13+, `uv` only (not pip)
- ruff: line-length 100, double quotes, force-single-line imports
- Modern type syntax: `dict[str, Any]`, `Path | None` (no `typing` module)
- Always `pathlib.Path`, never `os.path`
- CLI scripts use `typer`
