import tomllib
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader import SO101LeaderConfig

CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


config = load_config()

# Left arm configuration
left_leader_config = config["leader"]["left"]
left_follower_config = config["follower"]["left"]

# Right arm configuration
right_leader_config = config["leader"]["right"]
right_follower_config = config["follower"]["right"]

# Validate all ports are configured
for arm in ["left", "right"]:
    for role in ["leader", "follower"]:
        port = config[role][arm]["port"]
        if not port:
            msg = f"Error: No port configured for {arm} {role}. Please update ports.toml."
            raise ValueError(msg)

# Create robot (follower) configurations
left_robot_config = SO101FollowerConfig(
    port=left_follower_config["port"],
    id=left_follower_config["id"],
)

right_robot_config = SO101FollowerConfig(
    port=right_follower_config["port"],
    id=right_follower_config["id"],
)

# Create teleop (leader) configurations
left_teleop_config = SO101LeaderConfig(
    port=left_leader_config["port"],
    id=left_leader_config["id"],
)

right_teleop_config = SO101LeaderConfig(
    port=right_leader_config["port"],
    id=right_leader_config["id"],
)

# Create robot and teleop instances
left_robot = SO101Follower(left_robot_config)
right_robot = SO101Follower(right_robot_config)
left_teleop = SO101Leader(left_teleop_config)
right_teleop = SO101Leader(right_teleop_config)

# Connect all devices (skip calibration check - assumes already calibrated)
print("Connecting left leader...")
left_teleop.connect(calibrate=False)
print("Connecting right leader...")
right_teleop.connect(calibrate=False)
print("Connecting left follower...")
left_robot.connect(calibrate=False)
print("Connecting right follower...")
right_robot.connect(calibrate=False)
print("All devices connected!\n")


def print_dual_table(
    left_leader: dict,
    left_follower: dict,
    right_leader: dict,
    right_follower: dict,
) -> None:
    """Print a table of all leader and follower positions for both arms."""
    # Get all joint keys (should be the same for all)
    keys = sorted(set(left_leader.keys()) | set(left_follower.keys()))

    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")

    # Header
    print(f"{'Joint':<20} {'L-Leader':>10} {'L-Follow':>10} {'R-Leader':>10} {'R-Follow':>10}")
    print("-" * 62)

    for key in keys:
        left_lead_val = left_leader.get(key, "")
        left_follow_val = left_follower.get(key, "")
        right_lead_val = right_leader.get(key, "")
        right_follow_val = right_follower.get(key, "")

        def format_val(val: float | str) -> str:
            if isinstance(val, float):
                return f"{val:>10.2f}"
            return f"{val!s:>10}"

        print(
            f"{key:<20} "
            f"{format_val(left_lead_val)} "
            f"{format_val(left_follow_val)} "
            f"{format_val(right_lead_val)} "
            f"{format_val(right_follow_val)}"
        )


while True:
    # Get leader positions
    left_leader_pos = left_teleop.get_action()
    right_leader_pos = right_teleop.get_action()

    # Send to followers
    left_robot.send_action(left_leader_pos)
    right_robot.send_action(right_leader_pos)

    # Get follower observations
    left_follower_pos = left_robot.get_observation()
    right_follower_pos = right_robot.get_observation()

    # Display all positions
    print_dual_table(left_leader_pos, left_follower_pos, right_leader_pos, right_follower_pos)
