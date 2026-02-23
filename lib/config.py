"""
Configuration loading and validation utilities.
"""

import subprocess
import tomllib
from pathlib import Path

from huggingface_hub import HfApi

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"
CALIBRATION_DIR = ROOT_DIR / "calibration"

# Data directory for all local data storage (gitignored)
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RECORDINGS_DIR = DATA_DIR / "recordings"
OUTPUTS_DIR = DATA_DIR / "outputs"


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


def validate_config(config: dict) -> None:
    """Validate that all required hardware ports are configured."""
    for arm in ["left", "right"]:
        for role in ["leader", "follower"]:
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
        return {
            "path": cam_cfg["path"],
            "width": cam_cfg["width"],
            "height": cam_cfg["height"],
            "fps": cam_cfg["fps"],
            "fourcc": cam_cfg.get("fourcc"),
        }

    # If specific camera requested
    if camera_name:
        return get_single_camera(camera_section[camera_name])

    # Return all cameras
    cameras = {}
    for name in ["top", "left", "right"]:
        if name in camera_section:
            cameras[name] = get_single_camera(camera_section[name])

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
