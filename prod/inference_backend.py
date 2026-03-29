"""
Inference backend abstraction for local and remote policy execution.
"""

from typing import Protocol

from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

from lib.policy import load_policy_stack


class InferenceBackend(Protocol):
    """Protocol for inference backends (local GPU or remote server)."""

    def load(self, model_config: dict) -> None: ...
    def predict(self, observation: dict, robot_name: str) -> dict: ...


class LocalInference:
    """Loads and runs policy inference on the local machine."""

    def __init__(self):
        self._policy = None
        self._policy_cfg = None
        self._preprocessor = None
        self._postprocessor = None
        self._dataset = None
        self._robot_action_processor = None
        self._robot_obs_processor = None
        self._device = None
        self._task = ""

    def load(self, model_config: dict) -> None:
        """Load policy stack from model config."""
        (
            self._policy,
            self._policy_cfg,
            self._preprocessor,
            self._postprocessor,
            self._dataset,
            self._robot_action_processor,
            self._robot_obs_processor,
        ) = load_policy_stack(
            policy_repo_id=model_config["policy_repo_id"],
            dataset_repo_id=model_config["dataset_repo_id"],
            device=model_config.get("device", "cuda"),
            temporal_ensemble_coeff=model_config.get("temporal_ensemble_coeff"),
        )
        self._device = get_safe_torch_device(model_config.get("device", "cuda"))
        self._task = model_config.get("task", "")

    def predict(self, observation: dict, robot_name: str) -> dict:
        """Run policy inference on a single observation and return the action."""
        obs_processed = self._robot_obs_processor(observation)
        observation_frame = build_dataset_frame(
            self._dataset.features,
            obs_processed,
            prefix=OBS_STR,
        )
        action_values = predict_action(
            observation=observation_frame,
            policy=self._policy,
            device=self._device,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            use_amp=getattr(self._policy_cfg, "use_amp", False),
            task=self._task,
            robot_type=robot_name,
        )
        robot_action = make_robot_action(action_values, self._dataset.features)
        return self._robot_action_processor((robot_action, observation))


class RemoteInference:
    """Sends observations to a remote inference server (Phase 5)."""

    def __init__(self):
        self._server_url = ""
        self._task = ""

    def load(self, model_config: dict) -> None:
        self._server_url = model_config["server_url"]
        self._task = model_config.get("task", "")

    def predict(self, observation: dict, robot_name: str) -> dict:
        raise NotImplementedError("Remote inference not yet implemented (Phase 5)")
