"""
Shared library for robot configuration and utilities.
"""

from lib.config import get_camera_config
from lib.config import get_recording_config
from lib.config import load_config
from lib.config import validate_config

__all__ = [
    "get_camera_config",
    "get_recording_config",
    "load_config",
    "validate_config",
]
