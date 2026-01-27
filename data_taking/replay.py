"""
Replay an episode from a dataset on the bimanual SO101 robot arms.

Usage:
    uv run data_taking/replay.py --episode 0
    uv run data_taking/replay.py --episode 0 --repo-id jhimmens/linique
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from lib.config import get_recording_config
from lib.config import load_config

app = typer.Typer()

LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"


@app.command()
def main(
    episode: int = typer.Option(
        ...,
        "--episode",
        "-e",
        help="Episode number to replay",
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        "-r",
        help="Dataset repository ID (defaults to config value)",
    ),
    fps: int = typer.Option(
        30,
        "--fps",
        help="Playback frames per second",
    ),
    play_sounds: bool = typer.Option(
        False,
        "--sounds/--no-sounds",
        help="Play audio cues",
    ),
) -> None:
    """Replay a recorded episode on the bimanual robot."""
    typer.echo(f"\n=== Replay Episode {episode} ===\n")

    # Load configs
    config = load_config()
    local_config = load_config(LOCAL_CONFIG_PATH)
    config.update(local_config)

    recording_cfg = get_recording_config(config)

    # Use provided repo_id or fall back to config
    dataset_repo_id = repo_id or recording_cfg["repo_id"]

    left_follower_cfg = config["follower"]["left"]
    right_follower_cfg = config["follower"]["right"]

    typer.echo(f"Dataset:        {dataset_repo_id}")
    typer.echo(f"Episode:        {episode}")
    typer.echo(f"Left Follower:  {left_follower_cfg['port']}")
    typer.echo(f"Right Follower: {right_follower_cfg['port']}")

    if not typer.confirm("\nProceed with replay?"):
        typer.echo("Replay cancelled.")
        raise typer.Exit(0)

    # Create robot config (no cameras needed for replay)
    robot_config = BiSOFollowerConfig(
        id="bimanual_follower",
        left_arm_config=SOFollowerConfig(
            port=left_follower_cfg["port"],
        ),
        right_arm_config=SOFollowerConfig(
            port=right_follower_cfg["port"],
        ),
    )

    # Load dataset and filter to requested episode
    typer.echo(f"\nLoading dataset episode {episode}...")
    dataset = LeRobotDataset(dataset_repo_id, episodes=[episode])

    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    actions = episode_frames.select_columns(ACTION)
    num_frames = len(episode_frames)

    typer.echo(f"Loaded {num_frames} frames")

    # Initialize robot
    typer.echo("Connecting robot...")
    robot = BiSOFollower(robot_config)
    robot_action_processor = make_default_robot_action_processor()

    try:
        robot.connect()

        log_say(f"Replaying episode {episode}", play_sounds, blocking=True)

        for idx in range(num_frames):
            start_t = time.perf_counter()

            # Get action from dataset
            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            # Process and send action
            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            # Maintain timing
            dt_s = time.perf_counter() - start_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            # Progress indicator every 30 frames
            if idx % 30 == 0:
                typer.echo(f"  Frame {idx}/{num_frames}")

        log_say("Replay complete", play_sounds, blocking=True)

    finally:
        if robot.is_connected:
            typer.echo("Disconnecting robot...")
            robot.disconnect()

    typer.echo("\nReplay complete!")


if __name__ == "__main__":
    app()
