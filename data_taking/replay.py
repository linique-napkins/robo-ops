"""
Replay an episode from a dataset on the bimanual SO101 robot arms.

Usage:
    uv run data_taking/replay.py --episode 0
    uv run data_taking/replay.py --episode 0 --repo-id jhimmens/linique
    uv run data_taking/replay.py --episode 0 --display  # With Rerun + URDF visualization
    uv run data_taking/replay.py --episode 0 --display --no-arms  # Visualization only, no robot
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from lib.config import get_local_dataset_path
from lib.config import get_recording_config
from lib.config import get_urdf_config
from lib.config import load_config
from lib.robots import get_bimanual_follower
from lib.stow import stow_and_disconnect
from lib.urdf_viz import init_rerun_with_urdf
from lib.urdf_viz import log_dataset_images
from lib.urdf_viz import log_observation_and_action
from lib.urdf_viz import save_rrd

app = typer.Typer()

LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"


def build_observation_from_frame(frame: dict, feature_names: list[str]) -> dict:
    """Build an observation dict from a dataset frame.

    Args:
        frame: Dataset frame containing observation.state array.
        feature_names: List of feature names (e.g., ['left_shoulder_pan.pos', ...]).

    Returns:
        Dict with keys like 'left_shoulder_pan.pos' and float values.
    """
    state_array = frame["observation.state"]
    observation = {}
    for i, name in enumerate(feature_names):
        observation[name] = float(state_array[i])
    return observation


@app.command()
def main(  # noqa: PLR0912
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
    use_arms: bool = typer.Option(
        True,
        "--arms/--no-arms",
        help="Connect to robot arms (disable for visualization-only mode)",
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

    typer.echo(f"Dataset:        {dataset_repo_id}")
    typer.echo(f"Episode:        {episode}")
    if use_arms:
        typer.echo(f"Left Follower:  {config['follower']['left']['port']}")
        typer.echo(f"Right Follower: {config['follower']['right']['port']}")
    else:
        typer.echo("Mode:           Visualization only (no robot arms)")

    # Load dataset and filter to requested episode
    # Uses local data/datasets/ path, falling back to HuggingFace Hub download
    local_path = get_local_dataset_path(dataset_repo_id)
    typer.echo(f"\nLoading dataset episode {episode}...")
    dataset = LeRobotDataset(dataset_repo_id, root=local_path, episodes=[episode])

    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    num_frames = len(episode_frames)

    # Get image feature keys from dataset
    image_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    typer.echo(f"Found {len(image_keys)} camera(s): {image_keys}")

    typer.echo(f"Loaded {num_frames} frames")

    # Get state feature names for building observations from dataset
    state_feature_names = dataset.features.get("observation.state", {}).get("names", [])

    # Initialize robot if using arms
    robot = None
    robot_action_processor = None
    if use_arms:
        typer.echo("Connecting robot...")
        robot = get_bimanual_follower(config)
        robot_action_processor = make_default_robot_action_processor()

    # Initialize visualization (always enabled for replay)
    urdf_cfg = get_urdf_config(config)
    typer.echo("Initializing Rerun visualization with URDF...")

    # Extract camera names from dataset features
    # Keys are like 'observation.images.left_top_cam' -> extract 'top'
    camera_names = []
    for k in image_keys:
        cam_name = k.replace("observation.images.", "")
        # Remove _cam suffix
        if cam_name.endswith("_cam"):
            cam_name = cam_name[:-4]
        # Remove left_/right_ prefix (from BiSOFollower)
        if cam_name.startswith("left_"):
            cam_name = cam_name[5:]
        elif cam_name.startswith("right_"):
            cam_name = cam_name[6:]
        camera_names.append(cam_name)

    visualizer = init_rerun_with_urdf(
        session_name="replay",
        urdf_path=urdf_cfg["path"],
        left_offset=urdf_cfg["left_offset"],
        right_offset=urdf_cfg["right_offset"],
        left_rotation_deg=urdf_cfg["left_rotation"],
        right_rotation_deg=urdf_cfg["right_rotation"],
        camera_names=camera_names if camera_names else None,
    )
    if visualizer:
        typer.echo("URDF visualization initialized!")
    else:
        typer.echo("Warning: URDF visualization failed to initialize")

    try:
        if robot:
            robot.connect()

        log_say(f"Replaying episode {episode}", play_sounds, blocking=True)

        for idx in range(num_frames):
            start_t = time.perf_counter()

            # Get full frame from dataset (includes images)
            frame = episode_frames[idx]

            if use_arms and robot and robot_action_processor:
                # Get action from dataset
                action_array = frame[ACTION]
                action = {}
                for i, name in enumerate(dataset.features[ACTION]["names"]):
                    action[name] = action_array[i]

                # Process and send action
                robot_obs = robot.get_observation()
                processed_action = robot_action_processor((action, robot_obs))
                robot.send_action(processed_action)

                # Log to visualization
                log_observation_and_action(
                    visualizer=visualizer,
                    observation=robot_obs,
                    action=action,
                    use_degrees=True,
                )
            else:
                # Visualization-only mode: build observation from dataset
                observation = build_observation_from_frame(frame, state_feature_names)
                action_array = frame[ACTION]
                action = {}
                for i, name in enumerate(dataset.features[ACTION]["names"]):
                    action[name] = action_array[i]

                log_observation_and_action(
                    visualizer=visualizer,
                    observation=observation,
                    action=action,
                    use_degrees=True,
                )

            # Log dataset images (from recorded data)
            log_dataset_images(frame)

            # Maintain timing
            dt_s = time.perf_counter() - start_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            # Progress indicator every 30 frames
            if idx % 30 == 0:
                typer.echo(f"  Frame {idx}/{num_frames}")

        log_say("Replay complete", play_sounds, blocking=True)

    finally:
        if robot:
            stow_and_disconnect(robot)

        # Save Rerun recording
        rrd_path = save_rrd()
        if rrd_path:
            typer.echo(f"Rerun recording saved to: {rrd_path}")

    typer.echo("\nReplay complete!")


if __name__ == "__main__":
    app()
