"""
Shared policy loading utilities.

Extracts model loading logic for reuse by inference/run.py and prod/ server.
"""

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.utils.device_utils import get_safe_torch_device

from lib.config import get_local_dataset_path


def load_policy_stack(
    policy_repo_id: str,
    dataset_repo_id: str,
    device: str,
    temporal_ensemble_coeff: float | None = None,
    n_action_steps: int | None = None,
    rename_map: dict[str, str] | None = None,
) -> tuple:
    """Load trained policy, preprocessor, postprocessor, and dataset metadata.

    Args:
        policy_repo_id: Path to trained policy (local or HF Hub).
        dataset_repo_id: Dataset repo ID for feature definitions.
        device: Torch device string ("cuda", "cpu", "mps").
        temporal_ensemble_coeff: Temporal ensemble coefficient, or None to disable.
        rename_map: Optional mapping from robot observation keys to policy keys
                    (e.g. camera name remapping).

    Returns:
        Tuple of (policy, policy_cfg, preprocessor, postprocessor, dataset,
                  robot_action_processor, robot_observation_processor).
    """
    policy_cfg = PreTrainedConfig.from_pretrained(policy_repo_id)
    policy_cfg.pretrained_path = policy_repo_id

    if temporal_ensemble_coeff is not None:
        policy_cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy_cfg.n_action_steps = 1
    elif n_action_steps is not None:
        policy_cfg.n_action_steps = n_action_steps

    local_path = get_local_dataset_path(dataset_repo_id)
    dataset = LeRobotDataset(dataset_repo_id, root=local_path)

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map or {})
    policy.eval()
    policy.reset()
    policy.to(get_safe_torch_device(device))

    preprocessor_overrides = {}
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {
            "rename_map": rename_map,
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    return (
        policy,
        policy_cfg,
        preprocessor,
        postprocessor,
        dataset,
        robot_action_processor,
        robot_observation_processor,
    )
