"""
Robot and teleoperator factory functions.

Provides consistent configuration for all scripts using the bimanual SO101 setup.
Calibration files are stored in the repo at calibration/{role}/.
"""

from lerobot.cameras import CameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from lib.config import get_calibration_dir


def get_bimanual_follower(
    config: dict,
    cameras: dict[str, CameraConfig] | None = None,
) -> BiSOFollower:
    """Create a bimanual follower robot with repo calibration.

    Args:
        config: Global config dict (from load_config()).
        cameras: Optional camera configs to attach to the left arm.

    Returns:
        Configured BiSOFollower instance (not yet connected).
    """

    robot_config = BiSOFollowerConfig(
        id="bimanual_follower",
        calibration_dir=get_calibration_dir("follower"),
        left_arm_config=SOFollowerConfig(
            port=config["follower"]["left"]["port"],
            cameras=cameras or {},
        ),
        right_arm_config=SOFollowerConfig(
            port=config["follower"]["right"]["port"],
        ),
    )

    return BiSOFollower(robot_config)


def get_bimanual_leader(config: dict) -> BiSOLeader:
    """Create a bimanual leader teleoperator with repo calibration.

    Args:
        config: Global config dict (from load_config()).

    Returns:
        Configured BiSOLeader instance (not yet connected).
    """
    left_leader_cfg = config["leader"]["left"]
    right_leader_cfg = config["leader"]["right"]

    teleop_config = BiSOLeaderConfig(
        id="bimanual_leader",
        calibration_dir=get_calibration_dir("leader"),
        left_arm_config=SOLeaderConfig(
            port=left_leader_cfg["port"],
        ),
        right_arm_config=SOLeaderConfig(
            port=right_leader_cfg["port"],
        ),
    )

    return BiSOLeader(teleop_config)


def get_single_follower(config: dict, arm: str) -> SO101Follower:
    """Create a single follower arm with repo calibration.

    Uses bimanual naming convention for calibration file consistency.

    Args:
        config: Global config dict (from load_config()).
        arm: Which arm ('left' or 'right').

    Returns:
        Configured SO101Follower instance (not yet connected).
    """

    follower_config = SO101FollowerConfig(
        port=config["follower"][arm]["port"],
        id=f"bimanual_follower_{arm}",
        calibration_dir=get_calibration_dir("follower"),
    )

    return SO101Follower(follower_config)


def get_single_leader(config: dict, arm: str) -> SO101Leader:
    """Create a single leader arm with repo calibration.

    Uses bimanual naming convention for calibration file consistency.

    Args:
        config: Global config dict (from load_config()).
        arm: Which arm ('left' or 'right').

    Returns:
        Configured SO101Leader instance (not yet connected).
    """
    port = config["leader"][arm]["port"]
    calibration_id = f"bimanual_leader_{arm}"

    leader_config = SO101LeaderConfig(
        port=port,
        id=calibration_id,
        calibration_dir=get_calibration_dir("leader"),
    )

    return SO101Leader(leader_config)
