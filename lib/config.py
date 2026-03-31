"""
Configuration loading and validation utilities.
"""

import os
import subprocess
import tomllib
from pathlib import Path

from huggingface_hub import HfApi

# Username → environment mapping
_ENV_MAP: dict[str, str] = {
    "nvd": "local",
    "jhimmens": "sockeye",
}

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"
CALIBRATION_DIR = ROOT_DIR / "calibration"

# Data directory for all local data storage (gitignored)
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RECORDINGS_DIR = DATA_DIR / "recordings"
OUTPUTS_DIR = DATA_DIR / "outputs"


def get_environment() -> str:
    """Detect the current environment from OS username.

    Returns:
        Environment name (e.g. "local", "sockeye").

    Raises:
        RuntimeError: If username is not in the known environment map.
    """
    username = os.getenv("USER", "")
    env = _ENV_MAP.get(username)
    if env is None:
        known = ", ".join(f"{u!r} → {e!r}" for u, e in _ENV_MAP.items())
        msg = f"Unknown user {username!r}. Known environments: {known}"
        raise RuntimeError(msg)
    return env


def load_training_config(config_path: Path, section: str) -> dict:
    """Load a training config with environment-specific overrides.

    Reads [section] from the TOML file, detects the current environment,
    merges [section.env.{env}] into the base dict, and strips the "env" key.

    Args:
        config_path: Path to the TOML config file.
        section: Top-level TOML section name (e.g. "training", "sarm").

    Returns:
        Flat configuration dictionary with env-specific values merged in.

    Raises:
        RuntimeError: If the detected environment has no config section.
        KeyError: If the section is missing from the TOML file.
    """
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    base = dict(raw[section])
    env = get_environment()
    env_overrides = base.get("env", {}).get(env)
    if env_overrides is None:
        msg = (
            f"No [{section}.env.{env}] section in {config_path}. "
            f"Add environment-specific config for {env!r}."
        )
        raise RuntimeError(msg)

    base.pop("env", None)
    base.update(env_overrides)
    return base


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from a TOML file.

    Args:
        config_path: Path to config file. Defaults to root config.toml.

    Returns:
        Configuration dictionary.
    """
    path = config_path or CONFIG_PATH
    with path.open("rb") as f:
        return tomllib.load(f)


def validate_config(config: dict, roles: list[str] | None = None) -> None:
    """Validate that all required hardware ports are configured.

    Args:
        config: Full configuration dictionary.
        roles: Which roles to validate (e.g. ["follower"]). Defaults to all.
    """
    for arm in ["left", "right"]:
        for role in roles or ["leader", "follower"]:
            port = config.get(role, {}).get(arm, {}).get("port")
            if not port:
                msg = f"No port configured for {arm} {role}. Please update config.toml."
                raise ValueError(msg)


def get_camera_config(config: dict, camera_name: str | None = None) -> dict:
    """Get camera configuration.

    Args:
        config: Full configuration dictionary.
        camera_name: Name of camera to get config for (top, left, right).
                    If None, returns all cameras as a dict.

    Returns:
        Single camera config dict if camera_name specified,
        otherwise dict of all camera configs keyed by name.

    Raises:
        KeyError: If required camera fields are missing from config.
    """
    camera_section = config["camera"]

    def get_single_camera(cam_cfg: dict) -> dict:
        cam_type = cam_cfg.get("type", "opencv")
        base = {
            "type": cam_type,
            "width": cam_cfg["width"],
            "height": cam_cfg["height"],
            "fps": cam_cfg["fps"],
        }
        if cam_type == "realsense":
            base["serial_number"] = cam_cfg["serial_number"]
        else:
            base["path"] = cam_cfg["path"]
            base["fourcc"] = cam_cfg.get("fourcc")
        return base

    # If specific camera requested
    if camera_name:
        return get_single_camera(camera_section[camera_name])

    # Return all cameras — iterate dynamically, not hardcoded
    cameras = {}
    for name, cam_cfg in camera_section.items():
        cameras[name] = get_single_camera(cam_cfg)

    return cameras


def get_recording_config(config: dict) -> dict:
    """Get recording configuration.

    Raises:
        KeyError: If [recording] section or required keys are missing.
    """
    recording = config["recording"]
    return {
        "repo_id": recording["repo_id"],
        "task": recording["task"],
        "num_episodes": recording["num_episodes"],
        "episode_time": recording["episode_time"],
        "reset_time": recording["reset_time"],
        "idle_timeout": recording["idle_timeout"],
    }


def get_urdf_config(config: dict) -> dict:
    """Get URDF visualization configuration.

    Raises:
        KeyError: If [urdf] section or required keys are missing.
    """
    urdf = config["urdf"]
    return {
        "path": ROOT_DIR / urdf["path"],
        "left_offset": tuple(urdf["left_offset"]),
        "right_offset": tuple(urdf["right_offset"]),
        "left_rotation": urdf["left_rotation"],
        "right_rotation": urdf["right_rotation"],
    }


def get_stow_config(config: dict | None = None) -> dict:
    """Get stow configuration.

    Args:
        config: Full configuration dictionary. If None, loads from default config.

    Returns:
        Dictionary with stow settings.

    Raises:
        KeyError: If [stow] section or 'wait' key is missing from config.
    """
    if config is None:
        config = load_config()
    stow = config["stow"]
    return {
        "wait": stow["wait"],
    }


def get_calibration_dir(role: str) -> Path:
    """Get calibration directory for a role (leader or follower).

    Calibration files are stored in the repo at:
        calibration/leader/left.json
        calibration/leader/right.json
        calibration/follower/left.json
        calibration/follower/right.json

    Args:
        role: Either 'leader' or 'follower'.

    Returns:
        Path to the calibration directory for that role.
    """
    return CALIBRATION_DIR / role


def dataset_exists_on_hub(repo_id: str) -> bool:
    """Check if a dataset with actual data exists on Hugging Face Hub."""
    hub_api = HfApi()
    try:
        files = hub_api.list_repo_files(repo_id, repo_type="dataset")
        return "meta/info.json" in files
    except Exception:
        return False


def get_local_dataset_path(repo_id: str) -> Path:
    """Get the local path for a dataset.

    Datasets are stored in data/datasets/{repo_id} instead of the
    HuggingFace cache (~/.cache/huggingface/lerobot/).
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR / repo_id


def get_git_info() -> dict:
    """Get git repository information for reproducibility.

    Returns:
        Dictionary with git hash, branch, and dirty status.
        Returns empty values if not in a git repo.
    """
    try:
        # Get current commit hash
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT_DIR,
        ).stdout.strip()

        # Get short hash
        git_hash_short = git_hash[:8]

        # Get current branch
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT_DIR,
        ).stdout.strip()

        # Check if working directory is dirty
        dirty_check = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT_DIR,
        )
        is_dirty = len(dirty_check.stdout.strip()) > 0

        return {
            "git_hash": git_hash,
            "git_hash_short": git_hash_short,
            "git_branch": branch,
            "git_dirty": is_dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_hash": "",
            "git_hash_short": "",
            "git_branch": "",
            "git_dirty": False,
        }
