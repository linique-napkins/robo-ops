"""
Configuration loading and validation utilities.
"""

import subprocess
import tomllib
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"


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


def get_camera_config(config: dict) -> dict:
    """Get camera configuration with defaults."""
    camera = config.get("camera", {})
    return {
        "path": camera.get("path", 0),
        "width": camera.get("width", 640),
        "height": camera.get("height", 480),
        "fps": camera.get("fps", 30),
    }


def get_recording_config(config: dict) -> dict:
    """Get recording configuration with defaults."""
    recording = config.get("recording", {})
    return {
        "repo_id": recording.get("repo_id", "jhimmens/linique"),
        "task": recording.get("task", "folding"),
        "num_episodes": recording.get("num_episodes", 50),
        "episode_time": recording.get("episode_time", 60),
        "reset_time": recording.get("reset_time", 10),
    }


def dataset_exists_on_hub(repo_id: str) -> bool:
    """Check if a dataset with actual data exists on Hugging Face Hub."""
    hub_api = HfApi()
    try:
        files = hub_api.list_repo_files(repo_id, repo_type="dataset")
        return "meta/info.json" in files
    except Exception:
        return False


def get_local_dataset_path(repo_id: str) -> Path:
    """Get the local cache path for a dataset."""
    return HF_LEROBOT_HOME / repo_id


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
