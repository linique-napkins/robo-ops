"""
URDF visualization utilities for Rerun.

Provides functions to load and animate SO101 robot arm models in the Rerun viewer.
Supports bimanual (left/right) arm setups with configurable positioning.

This module manually parses URDF files and logs meshes/transforms to Rerun,
since rr.urdf is not available in rerun-sdk < 0.28.
"""

import datetime
import math
import os
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

# Default path to URDF file (relative to project root)
DEFAULT_URDF_PATH = (
    Path(__file__).parent.parent / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
)


@dataclass
class URDFJoint:
    """Represents a URDF joint."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limit_lower: float
    limit_upper: float


@dataclass
class URDFLink:
    """Represents a URDF link with visual geometry."""

    name: str
    visual_meshes: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    # Each tuple is (mesh_path, origin_xyz, origin_rpy)


def parse_origin(element: ET.Element | None) -> tuple[tuple, tuple]:
    """Parse origin element to get xyz and rpy."""
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    xyz_str = element.get("xyz", "0 0 0")
    rpy_str = element.get("rpy", "0 0 0")

    xyz = tuple(float(x) for x in xyz_str.split())
    rpy = tuple(float(x) for x in rpy_str.split())

    return xyz, rpy


def parse_axis(element: ET.Element | None) -> tuple[float, float, float]:
    """Parse axis element."""
    if element is None:
        return (0.0, 0.0, 1.0)

    axis_str = element.get("xyz", "0 0 1")
    return tuple(float(x) for x in axis_str.split())


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll-pitch-yaw to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


def axis_angle_to_quaternion(
    axis: tuple[float, float, float], angle: float
) -> tuple[float, float, float, float]:
    """Convert axis-angle to quaternion (x, y, z, w)."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)

    ax, ay, az = ax / norm, ay / norm, az / norm
    half_angle = angle / 2
    s = math.sin(half_angle)

    return (ax * s, ay * s, az * s, math.cos(half_angle))


def multiply_quaternions(q1: tuple, q2: tuple) -> tuple[float, float, float, float]:
    """Multiply two quaternions (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class URDFParser:
    """Simple URDF parser for extracting joints and links."""

    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path
        self.urdf_dir = urdf_path.parent
        self.joints: dict[str, URDFJoint] = {}
        self.links: dict[str, URDFLink] = {}
        self.robot_name = ""
        self.root_link = ""
        self._parse()

    def _parse(self) -> None:
        """Parse the URDF file."""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        self.robot_name = root.get("name", "robot")

        # Parse links
        for link_elem in root.findall("link"):
            link_name = link_elem.get("name", "")
            visuals = []

            for visual in link_elem.findall("visual"):
                origin = visual.find("origin")
                xyz, rpy = parse_origin(origin)

                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        mesh_path = mesh.get("filename", "")
                        visuals.append((mesh_path, xyz, rpy))

            self.links[link_name] = URDFLink(name=link_name, visual_meshes=visuals)

        # Parse joints and find root link
        child_links = set()
        for joint_elem in root.findall("joint"):
            joint_name = joint_elem.get("name", "")
            joint_type = joint_elem.get("type", "fixed")

            parent = joint_elem.find("parent")
            child = joint_elem.find("child")
            parent_link = parent.get("link", "") if parent is not None else ""
            child_link = child.get("link", "") if child is not None else ""

            child_links.add(child_link)

            origin = joint_elem.find("origin")
            xyz, rpy = parse_origin(origin)

            axis_elem = joint_elem.find("axis")
            axis = parse_axis(axis_elem)

            limit_elem = joint_elem.find("limit")
            limit_lower = (
                float(limit_elem.get("lower", "-3.14")) if limit_elem is not None else -3.14
            )
            limit_upper = float(limit_elem.get("upper", "3.14")) if limit_elem is not None else 3.14

            self.joints[joint_name] = URDFJoint(
                name=joint_name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )

        # Find root link (not a child of any joint)
        for link_name in self.links:
            if link_name not in child_links:
                self.root_link = link_name
                break

    def get_kinematic_chain(self) -> list[str]:
        """Get the order of links from root to end effector."""
        chain = [self.root_link]
        current = self.root_link

        while True:
            found_child = False
            for joint in self.joints.values():
                if joint.parent_link == current:
                    chain.append(joint.child_link)
                    current = joint.child_link
                    found_child = True
                    break
            if not found_child:
                break

        return chain


class BimanualURDFVisualizer:
    """Manages URDF visualization for two robot arms in Rerun."""

    def __init__(
        self,
        urdf_path: Path | str | None = None,
        left_offset: tuple[float, float, float] = (-0.2, 0.0, 0.0),
        right_offset: tuple[float, float, float] = (0.2, 0.0, 0.0),
        left_rotation_deg: float = 0.0,
        right_rotation_deg: float = 0.0,
    ):
        self.urdf_path = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.left_rotation_deg = left_rotation_deg
        self.right_rotation_deg = right_rotation_deg

        self.parser: URDFParser | None = None
        self._initialized = False
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        # Maps link name to full entity path for each arm (built during initialization)
        self._link_paths: dict[str, dict[str, str]] = {"left": {}, "right": {}}

    def _validate_urdf_path(self) -> bool:
        """Check if URDF file exists."""
        if not self.urdf_path.exists():
            print(f"Warning: URDF file not found at {self.urdf_path}")
            print("Make sure the SO-ARM100 submodule is initialized:")
            print("  git submodule update --init --recursive")
            return False
        return True

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh | None:
        """Load a mesh file, with caching."""
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path]

        full_path = self.urdf_path.parent / mesh_path
        if not full_path.exists():
            return None

        try:
            mesh = trimesh.load(str(full_path))
            self._mesh_cache[mesh_path] = mesh
            return mesh
        except Exception as e:
            print(f"Warning: Failed to load mesh {mesh_path}: {e}")
            return None

    def _log_link_meshes(self, entity_prefix: str, link: URDFLink) -> None:
        """Log all visual meshes for a link."""
        for i, (mesh_path, xyz, rpy) in enumerate(link.visual_meshes):
            mesh = self._load_mesh(mesh_path)
            if mesh is None or not hasattr(mesh, "vertices"):
                continue

            mesh_entity = f"{entity_prefix}/visual_{i}"

            # Log mesh transform (relative to link)
            quat = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
            rr.log(
                mesh_entity,
                rr.Transform3D(
                    translation=xyz,
                    rotation=rr.Quaternion(xyzw=quat),
                ),
                static=True,
            )

            # Log mesh geometry
            rr.log(
                mesh_entity,
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_colors=np.full(
                        (len(mesh.vertices), 4), [180, 180, 180, 255], dtype=np.uint8
                    ),
                ),
                static=True,
            )

    def _log_arm_structure(
        self, arm_prefix: str, offset: tuple, rotation_deg: float, arm_side: str
    ) -> None:
        """Log the static arm structure with proper nested kinematic hierarchy.

        This builds a proper parent-child entity path structure so transforms propagate correctly:
        world/left_arm/base_link/shoulder_link/upper_arm_link/...
        """
        # Log arm root transform
        root_quat = axis_angle_to_quaternion((0, 0, 1), math.radians(rotation_deg))
        rr.log(
            arm_prefix,
            rr.Transform3D(
                translation=offset,
                rotation=rr.Quaternion(xyzw=root_quat),
            ),
            static=True,
        )

        # Build mapping from link name -> parent link name
        link_to_parent: dict[str, str | None] = {}
        link_to_joint: dict[str, URDFJoint] = {}
        for joint in self.parser.joints.values():
            link_to_parent[joint.child_link] = joint.parent_link
            link_to_joint[joint.child_link] = joint

        # Root link has no parent
        link_to_parent[self.parser.root_link] = None

        # Build full entity path for each link (recursive traversal from root)
        link_paths = self._link_paths[arm_side]

        def get_link_path(link_name: str) -> str:
            """Recursively build the full entity path for a link."""
            if link_name in link_paths:
                return link_paths[link_name]

            parent = link_to_parent.get(link_name)
            if parent is None:
                # Root link - direct child of arm prefix
                path = f"{arm_prefix}/{link_name}"
            else:
                # Child link - nested under parent
                parent_path = get_link_path(parent)
                path = f"{parent_path}/{link_name}"

            link_paths[link_name] = path
            return path

        # Build paths for all links
        for link_name in self.parser.links:
            get_link_path(link_name)

        # Log root link meshes (no transform needed, it's at arm origin)
        root_link = self.parser.links.get(self.parser.root_link)
        root_path = link_paths[self.parser.root_link]
        if root_link:
            self._log_link_meshes(root_path, root_link)

        # Log each joint transform and child link meshes
        for joint in self.parser.joints.values():
            child_path = link_paths[joint.child_link]

            # Log joint transform (child relative to parent)
            quat = rpy_to_quaternion(joint.origin_rpy[0], joint.origin_rpy[1], joint.origin_rpy[2])
            rr.log(
                child_path,
                rr.Transform3D(
                    translation=joint.origin_xyz,
                    rotation=rr.Quaternion(xyzw=quat),
                    axis_length=0,  # Hide axes
                ),
                static=(joint.joint_type == "fixed"),
            )

            # Log child link meshes
            child_link = self.parser.links.get(joint.child_link)
            if child_link:
                self._log_link_meshes(child_path, child_link)

    def initialize(self) -> bool:
        """Load URDF files and set up the 3D scene in Rerun."""
        if self._initialized:
            return True

        if not self._validate_urdf_path():
            return False

        try:
            self.parser = URDFParser(self.urdf_path)
            print(f"Loaded URDF: {self.parser.robot_name}")
            print(f"Root link: {self.parser.root_link}")
            print(f"Joints: {list(self.parser.joints.keys())}")

            # Log left arm
            self._log_arm_structure(
                "world/left_arm", self.left_offset, self.left_rotation_deg, "left"
            )

            # Log right arm
            self._log_arm_structure(
                "world/right_arm", self.right_offset, self.right_rotation_deg, "right"
            )

            self._initialized = True
            return True

        except Exception as e:
            print(f"Error initializing URDF visualization: {e}")
            traceback.print_exc()
            return False

    def log_robot_state(
        self,
        observation: dict,
        use_degrees: bool = True,
    ) -> None:
        """Update both robot arm models with current joint positions."""
        if not self._initialized or self.parser is None:
            return

        self._update_arm_joints(observation, "left_", "left", use_degrees)
        self._update_arm_joints(observation, "right_", "right", use_degrees)

    def _update_arm_joints(
        self,
        observation: dict,
        obs_prefix: str,
        arm_side: str,
        use_degrees: bool,
    ) -> None:
        """Update joints for a single arm.

        Args:
            observation: Dict with keys like 'left_shoulder_pan.pos', 'right_gripper.pos', etc.
            obs_prefix: Prefix for observation keys ('left_' or 'right_')
            arm_side: Which arm ('left' or 'right') to look up paths in _link_paths
            use_degrees: Whether observation values are in degrees (True) or radians (False)
        """
        link_paths = self._link_paths.get(arm_side, {})
        if not link_paths:
            return

        for joint_name, joint in self.parser.joints.items():
            if joint.joint_type not in ("revolute", "continuous", "prismatic"):
                continue

            # Map URDF joint name to observation key
            # Try both with and without .pos suffix (lerobot uses 'left_shoulder_pan.pos')
            obs_key = f"{obs_prefix}{joint_name}.pos"
            if obs_key not in observation:
                # Fall back to without .pos suffix (for test scripts)
                obs_key = f"{obs_prefix}{joint_name}"
                if obs_key not in observation:
                    continue

            value = observation[obs_key]
            if value is None:
                continue

            value = float(value.item()) if hasattr(value, "item") else float(value)

            if use_degrees and joint.joint_type in ("revolute", "continuous"):
                value = math.radians(value)

            # Compute transform: origin rotation + joint rotation
            origin_quat = rpy_to_quaternion(
                joint.origin_rpy[0], joint.origin_rpy[1], joint.origin_rpy[2]
            )

            if joint.joint_type in ("revolute", "continuous"):
                joint_quat = axis_angle_to_quaternion(joint.axis, value)
                combined_quat = multiply_quaternions(origin_quat, joint_quat)
            else:
                combined_quat = origin_quat

            translation = joint.origin_xyz
            if joint.joint_type == "prismatic":
                translation = (
                    joint.origin_xyz[0] + joint.axis[0] * value,
                    joint.origin_xyz[1] + joint.axis[1] * value,
                    joint.origin_xyz[2] + joint.axis[2] * value,
                )

            # Update the child link transform using the nested entity path
            child_path = link_paths.get(joint.child_link)
            if not child_path:
                continue

            rr.log(
                child_path,
                rr.Transform3D(
                    translation=translation,
                    rotation=rr.Quaternion(xyzw=combined_quat),
                    axis_length=0,
                ),
            )


# Global visualizer instance
_global_visualizer: BimanualURDFVisualizer | None = None
_global_rrd_path: Path | None = None
_frame_counter = 0

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def reset_frame_counter() -> None:
    """Reset the frame counter (call at the start of each episode/session)."""
    global _frame_counter  # noqa: PLW0603
    _frame_counter = 0


def log_joint_scalars(
    observation: dict | None = None,
    action: dict | None = None,
) -> None:
    """Log joint positions as scalars for time-series overlay (actual vs commanded).

    Logs to entity paths:
        joints/{side}/{joint}/actual      — observation (blue)
        joints/{side}/{joint}/commanded   — action (orange)
    """
    for side in ["left", "right"]:
        for joint in JOINT_NAMES:
            key = f"{side}_{joint}.pos"
            if observation and key in observation:
                val = observation[key]
                val = float(val.item()) if hasattr(val, "item") else float(val)
                rr.log(f"joints/{side}/{joint}/actual", rr.Scalars(val))
            if action and key in action:
                val = action[key]
                val = float(val.item()) if hasattr(val, "item") else float(val)
                rr.log(f"joints/{side}/{joint}/commanded", rr.Scalars(val))


def get_global_visualizer() -> BimanualURDFVisualizer | None:
    """Get the global URDF visualizer instance."""
    return _global_visualizer


def get_rrd_path(session_name: str) -> Path:
    """Generate a unique .rrd file path for a session.

    Args:
        session_name: Name of the session (e.g., 'recording', 'teleop', 'replay').

    Returns:
        Path to the .rrd file (e.g., data/recordings/recording_20260203_153045.rrd).
    """
    from lib.config import RECORDINGS_DIR  # noqa: PLC0415

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return RECORDINGS_DIR / f"{session_name}_{timestamp}.rrd"


def save_rrd() -> Path | None:
    """Return the path to the .rrd file being written.

    The file sink is set up in init_rerun_with_urdf() so data is written
    continuously. This just returns the path for display purposes.
    """
    return _global_rrd_path


def init_rerun_with_urdf(
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    urdf_path: Path | str | None = None,
    left_offset: tuple[float, float, float] = (-0.2, 0.0, 0.0),
    right_offset: tuple[float, float, float] = (0.2, 0.0, 0.0),
    left_rotation_deg: float = 0.0,
    right_rotation_deg: float = 0.0,
    camera_names: list[str] | None = None,
) -> BimanualURDFVisualizer | None:
    """Initialize Rerun with URDF visualization for bimanual robot arms.

    Args:
        session_name: Name for the Rerun session.
        ip: Optional IP for remote Rerun server.
        port: Optional port for remote Rerun server.
        urdf_path: Path to URDF file for robot visualization.
        left_offset: 3D offset for left arm in world coordinates.
        right_offset: 3D offset for right arm in world coordinates.
        left_rotation_deg: Rotation around Z axis for left arm (degrees).
        right_rotation_deg: Rotation around Z axis for right arm (degrees).
        camera_names: List of camera names (e.g., ["top", "left", "right"]) for image views.
                     If None, defaults to ["top", "left", "right"].

    Returns:
        BimanualURDFVisualizer instance or None if initialization failed.
    """
    global _global_visualizer, _global_rrd_path  # noqa: PLW0603

    if camera_names is None:
        camera_names = []

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    # Generate .rrd path for saving
    _global_rrd_path = get_rrd_path(session_name)

    # Use recording_id so multiple sessions show up in the same viewer
    rr.init(session_name, recording_id=str(_global_rrd_path.stem))

    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")

    # Set up dual sinks: file + viewer. spawn(connect=False) launches the viewer
    # without replacing the recording sink, then set_sinks() attaches both.
    if ip and port:
        grpc_sink = rr.GrpcSink(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(connect=False, memory_limit=memory_limit)
        grpc_sink = rr.GrpcSink()

    rr.set_sinks(rr.FileSink(str(_global_rrd_path)), grpc_sink)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    visualizer = BimanualURDFVisualizer(
        urdf_path=urdf_path,
        left_offset=left_offset,
        right_offset=right_offset,
        left_rotation_deg=left_rotation_deg,
        right_rotation_deg=right_rotation_deg,
    )

    if visualizer.initialize():
        _global_visualizer = visualizer

        # Style joint series (static — only needs to be set once)
        for side in ["left", "right"]:
            for joint in JOINT_NAMES:
                rr.log(
                    f"joints/{side}/{joint}/actual",
                    rr.SeriesLines(colors=[[0, 120, 255]], names=["actual"]),
                    static=True,
                )
                rr.log(
                    f"joints/{side}/{joint}/commanded",
                    rr.SeriesLines(colors=[[255, 80, 0]], names=["commanded"]),
                    static=True,
                )

        # Build camera image views
        camera_views = [
            rrb.Spatial2DView(name=f"{name.capitalize()} Camera", origin=f"cameras/{name}")
            for name in camera_names
        ]

        # Per-joint time series views (6 per arm = 12 total)
        left_joint_views = [
            rrb.TimeSeriesView(name=f"L {j}", origin=f"joints/left/{j}") for j in JOINT_NAMES
        ]
        right_joint_views = [
            rrb.TimeSeriesView(name=f"R {j}", origin=f"joints/right/{j}") for j in JOINT_NAMES
        ]

        # Layout: 3D URDF on left, cameras + joint plots on right
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="Robot Arms", origin="world"),
            rrb.Vertical(
                rrb.Grid(*camera_views, grid_columns=2) if camera_views else rrb.TextDocumentView(),
                rrb.Horizontal(
                    rrb.Vertical(*left_joint_views),
                    rrb.Vertical(*right_joint_views),
                ),
                row_shares=[2, 3],
            ),
            column_shares=[1, 2],
        )
        rr.send_blueprint(blueprint)

        return visualizer

    return None


def log_camera_images(observation: dict | None = None) -> None:
    """Log camera images from observation to Rerun.

    Handles camera keys in these formats:
    - '{name}_cam' (e.g., 'top_cam') -> logs to 'cameras/{name}'
    - 'left_{name}_cam' (e.g., 'left_top_cam' from BiSOFollower) -> logs to 'cameras/{name}'

    Args:
        observation: Dictionary containing camera images as numpy arrays.
    """
    if observation is None:
        return

    for key, value in observation.items():
        # Look for camera keys ending in '_cam'
        if not key.endswith("_cam"):
            continue

        if value is None:
            continue

        if not isinstance(value, np.ndarray):
            continue

        # Extract camera name (e.g., 'top_cam' -> 'top', 'left_top_cam' -> 'top')
        camera_name = key[:-4]  # Remove '_cam' suffix

        # Remove 'left_' or 'right_' prefix if present (from BiSOFollower)
        if camera_name.startswith("left_"):
            camera_name = camera_name[5:]
        elif camera_name.startswith("right_"):
            camera_name = camera_name[6:]

        # Convert CHW -> HWC if needed
        arr = value
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Log to cameras/{name} path
        rr.log(f"cameras/{camera_name}", rr.Image(arr))


def log_dataset_images(frame: dict) -> None:
    """Log camera images from a dataset frame to Rerun.

    Handles dataset image keys in these formats:
    - 'observation.images.{name}_cam' -> logs to 'cameras/{name}'
    - 'observation.images.left_{name}_cam' (from BiSOFollower) -> logs to 'cameras/{name}'

    Args:
        frame: A dictionary containing dataset frame data, including image keys.
    """
    if frame is None:
        return

    for key, value in frame.items():
        # Look for dataset image keys (observation.images.{name}_cam)
        if not key.startswith("observation.images."):
            continue

        if value is None:
            continue

        # Extract camera name from key (e.g., 'observation.images.left_top_cam' -> 'top')
        camera_key = key.replace("observation.images.", "")
        camera_name = camera_key[:-4] if camera_key.endswith("_cam") else camera_key

        # Remove 'left_' or 'right_' prefix if present (from BiSOFollower)
        if camera_name.startswith("left_"):
            camera_name = camera_name[5:]
        elif camera_name.startswith("right_"):
            camera_name = camera_name[6:]

        # Convert to numpy array if needed (handles torch tensors too)
        if hasattr(value, "numpy"):
            arr = value.numpy()
        elif not isinstance(value, np.ndarray):
            arr = np.array(value)
        else:
            arr = value

        # Convert CHW -> HWC if needed
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Log to cameras/{name} path
        rr.log(f"cameras/{camera_name}", rr.Image(arr))


def log_observation_and_action(
    visualizer: BimanualURDFVisualizer | None,
    observation: dict | None = None,
    action: dict | None = None,
    use_degrees: bool = True,
) -> None:
    """Log observation and action data to Rerun, including URDF joint updates and camera images."""
    global _frame_counter  # noqa: PLW0603

    rr.set_time("step", sequence=_frame_counter)
    rr.set_time("wall_clock", timestamp=time.time())
    _frame_counter += 1

    if visualizer is not None and observation is not None:
        visualizer.log_robot_state(observation, use_degrees=use_degrees)

    log_camera_images(observation)
    log_joint_scalars(observation=observation, action=action)


def log_urdf_state(observation: dict | None = None, use_degrees: bool = True) -> None:
    """Update global URDF visualizer with observation."""
    if _global_visualizer is not None and observation is not None:
        _global_visualizer.log_robot_state(observation, use_degrees=use_degrees)
