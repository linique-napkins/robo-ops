"""
Shared library for robot configuration and utilities.
"""

from lib.config import CALIBRATION_DIR
from lib.config import DATA_DIR
from lib.config import DATASETS_DIR
from lib.config import OUTPUTS_DIR
from lib.config import RECORDINGS_DIR
from lib.config import ROOT_DIR
from lib.config import get_calibration_dir
from lib.config import get_camera_config
from lib.config import get_local_dataset_path
from lib.config import get_recording_config
from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader
from lib.robots import get_single_follower
from lib.robots import get_single_leader
from lib.stow import stow
from lib.stow import stow_and_disconnect

__all__ = [
    "CALIBRATION_DIR",
    "DATASETS_DIR",
    "DATA_DIR",
    "OUTPUTS_DIR",
    "RECORDINGS_DIR",
    "ROOT_DIR",
    "get_bimanual_follower",
    "get_bimanual_leader",
    "get_calibration_dir",
    "get_camera_config",
    "get_local_dataset_path",
    "get_recording_config",
    "get_single_follower",
    "get_single_leader",
    "load_config",
    "stow",
    "stow_and_disconnect",
    "validate_config",
]
