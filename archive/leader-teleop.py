import tomllib
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader import SO101LeaderConfig

# Config: which arm to use ("left" or "right")
ARM = "right"

CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


config = load_config()

leader_config = config["leader"][ARM]
follower_config = config["follower"][ARM]

robot_config = SO101FollowerConfig(
    port=follower_config["port"],
    id=follower_config["id"],
)

teleop_config = SO101LeaderConfig(
    port=leader_config["port"],
    id=leader_config["id"],
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()


def print_table(leader: dict, follower: dict) -> None:
    """Print a table of leader and follower positions, updating in place."""
    keys = sorted(set(leader.keys()) | set(follower.keys()))

    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")

    # Header
    print(f"{'Joint':<20} {'Leader':>12} {'Follower':>12}")
    print("-" * 46)

    for key in keys:
        leader_val = leader.get(key, "")
        follower_val = follower.get(key, "")

        if isinstance(leader_val, float):
            leader_str = f"{leader_val:>12.2f}"
        else:
            leader_str = f"{leader_val!s:>12}"

        if isinstance(follower_val, float):
            follower_str = f"{follower_val:>12.2f}"
        else:
            follower_str = f"{follower_val!s:>12}"

        print(f"{key:<20} {leader_str} {follower_str}")


while True:
    leader_pos = teleop_device.get_action()
    robot.send_action(leader_pos)
    follower_pos = robot.get_observation()
    print_table(leader_pos, follower_pos)
