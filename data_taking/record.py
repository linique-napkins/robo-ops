"""
Data collection script for bimanual SO101 robot arms with a single camera.

Records teleoperation data from 2 leader/follower arm pairs and 1 camera
for training robot manipulation policies.

Usage:
    uv run data_taking/record.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

# Suppress noisy library logs before imports trigger them
os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1: only warnings+errors (not info spam)
os.environ.setdefault("RUST_LOG", "warn")  # Rerun: only warnings+errors

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

from lib.config import get_camera_config
from lib.config import get_git_info
from lib.config import get_local_dataset_path
from lib.config import get_recording_config
from lib.config import get_urdf_config
from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader
from lib.stow import stow_and_disconnect
from lib.urdf_viz import init_rerun_with_urdf
from lib.urdf_viz import log_camera_images
from lib.urdf_viz import log_urdf_state
from lib.urdf_viz import save_rrd

app = typer.Typer()

# Local config for this module
LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"

# Joint position change threshold (degrees/frame) below which arms are considered idle
IDLE_THRESHOLD = 0.15

# stderr fd backup for suppressing native C library noise
_stderr_backup_fd: int | None = None
_devnull_fd: int | None = None


def _suppress_native_logs() -> None:
    """Redirect stderr at the fd level to suppress libjpeg/ffmpeg C library noise.

    These libraries (libjpeg "Corrupt JPEG data", ffmpeg "[mp4 @]") print directly
    to stderr from C code, bypassing Python's logging entirely.
    """
    global _stderr_backup_fd, _devnull_fd  # noqa: PLW0603
    _stderr_backup_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)


def _restore_stderr() -> None:
    """Restore stderr so Python errors/tracebacks are visible again."""
    global _stderr_backup_fd, _devnull_fd  # noqa: PLW0603
    if _stderr_backup_fd is not None:
        os.dup2(_stderr_backup_fd, 2)
        os.close(_stderr_backup_fd)
        _stderr_backup_fd = None
    if _devnull_fd is not None:
        os.close(_devnull_fd)
        _devnull_fd = None


def _log_banner(msg: str) -> None:
    """Print a highly visible banner to stdout."""
    width = max(len(msg) + 4, 60)
    bar = "=" * width
    typer.echo(f"\n{bar}")
    typer.echo(f"  {msg}")
    typer.echo(f"{bar}\n")


def _countdown(
    reset_time: float,
    robot,
    events: dict,
    fps: int,
    teleop,
    display: bool,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    task: str,
) -> None:
    """Run the reset record_loop with a countdown timer on a single line."""
    import threading  # noqa: PLC0415

    stop_event = threading.Event()
    countdown_seconds = int(reset_time)

    def _print_countdown() -> None:
        remaining = countdown_seconds
        while remaining > 0 and not stop_event.is_set():
            print(f"\r  Reset: {remaining}s remaining...  ", end="", flush=True)
            stop_event.wait(timeout=1.0)
            remaining -= 1
        print("\r  Reset: done!                  ", flush=True)

    countdown_thread = threading.Thread(target=_print_countdown, daemon=True)
    countdown_thread.start()

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

    stop_event.set()
    countdown_thread.join()


class IdleDetector:
    """Detects when robot arms stop moving and triggers early exit."""

    def __init__(self, timeout: float, events: dict) -> None:
        self.timeout = timeout
        self.events = events
        self.enabled = False
        self._prev_positions: dict[str, float] = {}
        self._last_movement_time: float = 0.0

    def reset(self) -> None:
        self._prev_positions.clear()
        self._last_movement_time = time.monotonic()

    def update(self, observation: dict) -> None:
        if not self.enabled or self.timeout <= 0:
            return

        now = time.monotonic()
        moved = False

        for key, value in observation.items():
            if not key.endswith(".pos"):
                continue
            if not isinstance(value, (int, float)):
                continue
            if (
                key in self._prev_positions
                and abs(value - self._prev_positions[key]) > IDLE_THRESHOLD
            ):
                moved = True
            self._prev_positions[key] = value

        if moved or not self._prev_positions:
            self._last_movement_time = now
            return

        idle_duration = now - self._last_movement_time
        if idle_duration >= self.timeout:
            log_say("No movement detected. Ending episode.", play_sounds=True)
            self.events["exit_early"] = True


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
    idle = recording_cfg["idle_timeout"]
    typer.echo(f"  Idle timeout:   {idle}s" if idle > 0 else "  Idle timeout:   disabled")


def is_valid_local_dataset(local_path: Path) -> bool:
    """Check if a local dataset directory contains a valid, complete dataset.

    A valid dataset must have:
    - meta/info.json
    - meta/tasks.json (or meta/tasks.jsonl)
    - At least one episode parquet file OR be empty (0 episodes)
    """
    meta_dir = local_path / "meta"
    if not meta_dir.exists():
        return False

    # Check required metadata files
    if not (meta_dir / "info.json").exists():
        return False

    # tasks.json is required for a complete dataset
    return (meta_dir / "tasks.json").exists() or (meta_dir / "tasks.jsonl").exists()


def create_or_resume_dataset(
    resume: bool,
    repo_id: str,
    fps: int,
    robot: BiSOFollower,
    dataset_features: dict,
) -> LeRobotDataset:
    """Create a new dataset or resume an existing one.

    Datasets are stored locally in data/datasets/{repo_id}.
    Only checks for local datasets - does not contact HuggingFace Hub.
    """
    local_path = get_local_dataset_path(repo_id)

    # Check for existing valid local dataset
    if resume and local_path.exists() and is_valid_local_dataset(local_path):
        typer.echo(f"Resuming local dataset: {local_path}")
        dataset = LeRobotDataset(repo_id, root=local_path)
        if robot.cameras:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        return dataset

    if resume and local_path.exists():
        typer.echo(f"Found incomplete dataset at {local_path}, removing...")
        shutil.rmtree(local_path)

    if resume:
        typer.echo("No valid local dataset found, creating new dataset...")
    else:
        typer.echo(f"Creating new dataset: {repo_id}")

    # Clean up stale local cache if it exists (from failed previous runs)
    if local_path.exists():
        typer.echo(f"Removing existing local data: {local_path}")
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
        False,
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
    # TODO: Fix JPEG corruption — Innomaker cameras share USB 2.0 bus with motor
    # controllers, causing MJPG frame truncation under load. Move cameras to a
    # separate USB controller (Bus 002) to resolve. Warnings suppressed for now.
    _suppress_native_logs()

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
    idle_timeout = recording_cfg["idle_timeout"]

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
            fourcc=cam["fourcc"],
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

        # Set up idle detection
        idle_detector: IdleDetector | None = None
        if idle_timeout > 0:
            if not display:
                typer.echo(
                    "Warning: idle detection requires --display to monitor joint positions. "
                    "Idle auto-stop is disabled."
                )
            else:
                idle_detector = IdleDetector(timeout=idle_timeout, events=events)

        if display:
            urdf_cfg = get_urdf_config(config)
            init_rerun_with_urdf(
                session_name="recording",
                urdf_path=urdf_cfg["path"],
                left_offset=urdf_cfg["left_offset"],
                right_offset=urdf_cfg["right_offset"],
                left_rotation_deg=urdf_cfg["left_rotation"],
                right_rotation_deg=urdf_cfg["right_rotation"],
                camera_names=list(cameras_cfg.keys()),
            )

            # Patch record_loop's log function to also feed URDF viz + camera views
            import lerobot.scripts.lerobot_record as _record_mod  # noqa: PLC0415

            _orig_log = _record_mod.log_rerun_data

            def _patched_log(observation=None, action=None, **kwargs):
                _orig_log(observation=observation, action=action, **kwargs)
                if observation:
                    log_urdf_state(observation)
                    log_camera_images(observation)
                    if idle_detector:
                        idle_detector.update(observation)

            _record_mod.log_rerun_data = _patched_log

        typer.echo("\n=== Starting Recording ===")
        typer.echo("Controls:")
        typer.echo("  - Right Arrow: exit current episode early")
        typer.echo("  - Left Arrow:  re-record current episode")
        typer.echo("  - ESC:         stop recording\n")

        log_say("Ready to record. Press any key on the leader arms to begin.", play_sounds=True)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                episode_num = dataset.num_episodes
                remaining = num_episodes - recorded_episodes

                _log_banner(f"RECORDING  Episode {episode_num}  ({remaining} remaining)")
                log_say(
                    f"Recording episode {episode_num}. {remaining} episodes remaining.",
                    play_sounds=True,
                )

                # Enable idle detection for the episode
                if idle_detector:
                    idle_detector.reset()
                    idle_detector.enabled = True

                episode_start = time.monotonic()

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

                episode_duration = time.monotonic() - episode_start

                # Disable idle detection during save/reset
                if idle_detector:
                    idle_detector.enabled = False

                # Check for re-record before saving
                if events["rerecord_episode"]:
                    _log_banner("DISCARDED  Re-recording episode")
                    log_say("Discarding episode. Try again.", play_sounds=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Save the episode
                _log_banner(f"SAVING  Episode {episode_num}  ({episode_duration:.1f}s recorded)")
                save_start = time.monotonic()
                dataset.save_episode()
                save_duration = time.monotonic() - save_start
                recorded_episodes += 1
                remaining = num_episodes - recorded_episodes

                _log_banner(
                    f"SAVED  Episode {episode_num}  "
                    f"({episode_duration:.1f}s recorded, {save_duration:.1f}s to save, "
                    f"{remaining} remaining)"
                )
                log_say(
                    f"Episode {episode_num} saved. {remaining} remaining.",
                    play_sounds=True,
                )

                # Reset environment (skip for last episode)
                if not events["stop_recording"] and recorded_episodes < num_episodes:
                    log_say(
                        f"Reset the environment. {reset_time} seconds.",
                        play_sounds=True,
                    )
                    _countdown(
                        reset_time,
                        robot,
                        events,
                        fps,
                        teleop,
                        display,
                        teleop_action_processor,
                        robot_action_processor,
                        robot_observation_processor,
                        task,
                    )

        _log_banner(f"DONE  {recorded_episodes} episodes collected")
        log_say(
            f"Recording complete. {recorded_episodes} episodes collected.",
            play_sounds=True,
            blocking=True,
        )

    finally:
        _restore_stderr()

        if dataset:
            dataset.finalize()

        if teleop.is_connected:
            typer.echo("Disconnecting teleoperator...")
            teleop.disconnect()

        stow_and_disconnect(robot)

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
