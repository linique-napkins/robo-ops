"""
Data collection script for bimanual SO101 robot arms with a single camera.

Records teleoperation data from 2 leader/follower arm pairs and 1 camera
for training robot manipulation policies.

Usage:
    uv run data_taking/record.py
"""

import shutil
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features
from lerobot.datasets.pipeline_features import create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

from lib.config import dataset_exists_on_hub
from lib.config import get_camera_config
from lib.config import get_git_info
from lib.config import get_local_dataset_path
from lib.config import get_recording_config
from lib.config import get_urdf_config
from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader
from lib.urdf_viz import init_rerun_with_urdf
from lib.urdf_viz import save_rrd

app = typer.Typer()

# Local config for this module
LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"


def print_config(config: dict, recording_cfg: dict) -> None:
    """Print configuration summary."""
    cameras = get_camera_config(config)

    typer.echo("Configuration:")
    typer.echo(f"  Left Leader:    {config['leader']['left']['port']}")
    typer.echo(f"  Right Leader:   {config['leader']['right']['port']}")
    typer.echo(f"  Left Follower:  {config['follower']['left']['port']}")
    typer.echo(f"  Right Follower: {config['follower']['right']['port']}")
    typer.echo("  Cameras:")
    for name, cam in cameras.items():
        typer.echo(f"    {name}: {cam['path']} ({cam['width']}x{cam['height']} @ {cam['fps']}fps)")
    typer.echo("\nRecording settings:")
    typer.echo(f"  Repository:     {recording_cfg['repo_id']}")
    typer.echo(f"  Task:           {recording_cfg['task']}")
    typer.echo(f"  Episodes:       {recording_cfg['num_episodes']}")
    typer.echo(f"  Episode time:   {recording_cfg['episode_time']}s")
    typer.echo(f"  Reset time:     {recording_cfg['reset_time']}s")


def create_or_resume_dataset(
    resume: bool,
    repo_id: str,
    fps: int,
    robot: BiSOFollower,
    dataset_features: dict,
) -> LeRobotDataset:
    """Create a new dataset or resume an existing one.

    Datasets are stored locally in data/datasets/{repo_id}.
    When resuming, checks for local dataset first, then HuggingFace Hub.
    """
    local_path = get_local_dataset_path(repo_id)

    # Check for existing local dataset first
    if resume and local_path.exists():
        typer.echo(f"Resuming local dataset: {local_path}")
        dataset = LeRobotDataset(repo_id, root=local_path)
        if robot.cameras:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        return dataset

    # Check for existing dataset on HuggingFace Hub
    if resume and dataset_exists_on_hub(repo_id):
        typer.echo(f"Downloading and resuming dataset from Hub: {repo_id}")
        dataset = LeRobotDataset(repo_id, root=local_path)
        if robot.cameras:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        return dataset

    if resume:
        typer.echo(f"No existing dataset found at {repo_id}, creating new dataset...")
    else:
        typer.echo(f"Creating new dataset: {repo_id}")

    # Clean up stale local cache if it exists (from failed previous runs)
    if local_path.exists():
        typer.echo(f"Removing stale local cache: {local_path}")
        shutil.rmtree(local_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=local_path,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )


@app.command()
def main(  # noqa: PLR0912
    push_to_hub: bool = typer.Option(
        True,
        "--push/--no-push",
        help="Push dataset to Hugging Face Hub after recording",
    ),
    display: bool = typer.Option(
        True,
        "--display/--no-display",
        help="Display data in Rerun visualization",
    ),
    resume: bool = typer.Option(
        True,
        "--resume",
        help="Resume recording on an existing dataset",
    ),
) -> None:
    """Record teleoperation data from bimanual SO101 robot arms with a camera."""
    typer.echo("\n=== Bimanual Data Collection ===\n")

    # Load global config for hardware settings
    config = load_config()
    validate_config(config)

    # Load local config for recording settings, merge with global
    local_config = load_config(LOCAL_CONFIG_PATH)
    config.update(local_config)

    # Get camera and recording configuration
    cameras_cfg = get_camera_config(config)
    recording_cfg = get_recording_config(config)

    repo_id = recording_cfg["repo_id"]
    task = recording_cfg["task"]
    num_episodes = recording_cfg["num_episodes"]
    episode_time = recording_cfg["episode_time"]
    reset_time = recording_cfg["reset_time"]

    print_config(config, recording_cfg)

    # Get git info for reproducibility
    git_info = get_git_info()
    if git_info["git_hash"]:
        dirty_marker = " (dirty)" if git_info["git_dirty"] else ""
        typer.echo(f"\nGit: {git_info['git_hash_short']} on {git_info['git_branch']}{dirty_marker}")
        if git_info["git_dirty"]:
            typer.echo("  Warning: Working directory has uncommitted changes!")

    if not typer.confirm("\nProceed with recording?"):
        typer.echo("Recording cancelled.")
        raise typer.Exit(0)

    log_say("Starting data collection", play_sounds=True)

    # Build camera configs for robot
    camera_configs = {}
    for name, cam in cameras_cfg.items():
        camera_configs[f"{name}_cam"] = OpenCVCameraConfig(
            index_or_path=cam["path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )

    # Initialize robot and teleoperator using factory functions
    typer.echo("\nInitializing robot and teleoperator...")
    robot = get_bimanual_follower(config, cameras=camera_configs)
    teleop = get_bimanual_leader(config)

    # Create processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    # Build dataset features from robot and processor pipeline
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    dataset = None
    listener = None

    # Use FPS from first camera (they should all be the same)
    fps = next(iter(cameras_cfg.values()))["fps"]

    try:
        # Create or load dataset
        dataset = create_or_resume_dataset(resume, repo_id, fps, robot, dataset_features)

        # Add git info to dataset metadata for reproducibility
        if git_info["git_hash"]:
            dataset.meta.info["git_hash"] = git_info["git_hash"]
            dataset.meta.info["git_branch"] = git_info["git_branch"]
            dataset.meta.info["git_dirty"] = git_info["git_dirty"]
            typer.echo(f"Logged git hash: {git_info['git_hash_short']}")

        # Connect hardware
        typer.echo("Connecting robot...")
        robot.connect()
        typer.echo("Connecting teleoperator...")
        teleop.connect()

        # Initialize keyboard listener and visualization
        listener, events = init_keyboard_listener()
        if display:
            urdf_cfg = get_urdf_config(config)
            # Initialize rerun with URDF - visualizer is stored globally and used
            # by log_rerun_data via log_urdf_state callback
            init_rerun_with_urdf(
                session_name="recording",
                urdf_path=urdf_cfg["path"],
                left_offset=urdf_cfg["left_offset"],
                right_offset=urdf_cfg["right_offset"],
                left_rotation_deg=urdf_cfg["left_rotation"],
                right_rotation_deg=urdf_cfg["right_rotation"],
                camera_names=list(cameras_cfg.keys()),
            )

        typer.echo("\n=== Starting Recording ===")
        typer.echo("Controls:")
        typer.echo("  - Press 'q' to stop recording")
        typer.echo("  - Press 'r' to re-record current episode")
        typer.echo("  - Press 'e' to exit current episode early\n")

        log_say("Ready to record. Press any key on the leader arms to begin.", play_sounds=True)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                episode_num = dataset.num_episodes
                remaining = num_episodes - recorded_episodes
                log_say(
                    f"Recording episode {episode_num}. {remaining} episodes remaining.",
                    play_sounds=True,
                )

                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=episode_time,
                    single_task=task,
                    display_data=display,
                )

                # Check for re-record before saving
                if events["rerecord_episode"]:
                    log_say("Discarding episode. Try again.", play_sounds=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Save the episode
                dataset.save_episode()
                recorded_episodes += 1
                log_say(f"Episode {episode_num} saved.", play_sounds=True)

                # Reset environment (skip for last episode)
                if not events["stop_recording"] and recorded_episodes < num_episodes:
                    log_say(
                        f"Reset the environment. {reset_time} seconds.",
                        play_sounds=True,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=reset_time,
                        single_task=task,
                        display_data=display,
                    )

        log_say(
            f"Recording complete. {recorded_episodes} episodes collected.",
            play_sounds=True,
            blocking=True,
        )

    finally:
        if dataset:
            dataset.finalize()

        if robot.is_connected:
            typer.echo("Disconnecting robot...")
            robot.disconnect()

        if teleop.is_connected:
            typer.echo("Disconnecting teleoperator...")
            teleop.disconnect()

        if listener:
            listener.stop()

        if push_to_hub and dataset:
            log_say("Uploading dataset to hub.", play_sounds=True)
            typer.echo("\nPushing dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            log_say("Upload complete.", play_sounds=True)
            typer.echo("Dataset uploaded successfully!")

        # Save Rerun recording
        if display:
            rrd_path = save_rrd()
            if rrd_path:
                typer.echo(f"Rerun recording saved to: {rrd_path}")

    typer.echo("\nRecording complete!")


if __name__ == "__main__":
    app()
