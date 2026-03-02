"""
Training script for ACT policy on bimanual SO101 robot data.

Uses wandb for experiment tracking and pushes trained models to HuggingFace Hub.

Usage:
    uv run training/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import torch
import typer
import wandb
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from lib.config import OUTPUTS_DIR
from lib.config import get_local_dataset_path
from lib.config import load_config

app = typer.Typer()

LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert delta indices to timestamps."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def get_device(device_str: str) -> torch.device:
    """Get torch device, validating availability."""
    if device_str == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def print_training_config(training_cfg: dict) -> None:
    """Print training configuration summary."""
    typer.echo("\nTraining Configuration:")
    typer.echo(f"  Dataset:        {training_cfg['dataset_repo_id']}")
    typer.echo(f"  Policy:         {training_cfg['policy']}")
    typer.echo(f"  Output repo:    {training_cfg['output_repo_id']}")
    typer.echo(f"  Steps:          {training_cfg['steps']}")
    typer.echo(f"  Batch size:     {training_cfg['batch_size']}")
    typer.echo(f"  Learning rate:  {training_cfg['learning_rate']}")
    typer.echo(f"  Device:         {training_cfg['device']}")
    typer.echo(f"  Chunk size:     {training_cfg['chunk_size']}")
    typer.echo(f"  Output dir:     {training_cfg['output_dir'] or 'default (data/outputs/)'}")
    typer.echo(f"  Save freq:      every {training_cfg['save_freq']} steps")
    typer.echo(f"  Log freq:       every {training_cfg['log_freq']} steps")


def get_training_config(config: dict) -> dict:
    """Extract training configuration.

    Raises:
        KeyError: If [training] section or required keys are missing.
    """
    training = config["training"]
    return {
        "dataset_repo_id": training["dataset_repo_id"],
        "output_repo_id": training["output_repo_id"],
        "policy": training["policy"],
        "steps": training["steps"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "device": training["device"],
        "chunk_size": training["chunk_size"],
        "dim_model": training["dim_model"],
        "n_heads": training["n_heads"],
        "n_encoder_layers": training["n_encoder_layers"],
        "output_dir": training.get("output_dir"),
        "save_freq": training["save_freq"],
        "log_freq": training["log_freq"],
        "wandb_project": training["wandb_project"],
        "wandb_entity": training.get("wandb_entity"),
        "video_backend": training.get("video_backend"),
    }


@app.command()
def main(  # noqa: PLR0912
    push_to_hub: bool = typer.Option(
        False,
        "--push/--no-push",
        help="Push trained model to HuggingFace Hub",
    ),
    wandb_enabled: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging",
    ),
    resume_run_id: str = typer.Option(
        None,
        "--resume",
        help="Resume a previous wandb run by ID",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip confirmation prompts (for batch/SLURM jobs)",
    ),
    output_dir_override: str = typer.Option(
        None,
        "--output-dir",
        help="Override output directory for checkpoints (e.g. scratch on HPC)",
    ),
    steps_override: int = typer.Option(
        None,
        "--steps",
        help="Override training steps (e.g. --steps 5 for a quick test)",
    ),
) -> None:
    """Train ACT policy on recorded robot data."""
    typer.echo("\n=== ACT Policy Training ===\n")

    # Load training configuration
    config = load_config(LOCAL_CONFIG_PATH)
    training_cfg = get_training_config(config)

    print_training_config(training_cfg)

    if not yes and not typer.confirm("\nProceed with training?"):
        typer.echo("Training cancelled.")
        raise typer.Exit(0)

    # Setup device
    device = get_device(training_cfg["device"])
    typer.echo(f"\nUsing device: {device}")

    # Load dataset metadata and features
    # Uses local data/datasets/ path, falling back to HuggingFace Hub download
    dataset_repo_id = training_cfg["dataset_repo_id"]
    local_path = get_local_dataset_path(dataset_repo_id)
    typer.echo(f"\nLoading dataset: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=local_path)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    typer.echo(f"  Input features: {list(input_features.keys())}")
    typer.echo(f"  Output features: {list(output_features.keys())}")

    # Create ACT policy configuration
    typer.echo("\nCreating ACT policy...")
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=training_cfg["chunk_size"],
        n_action_steps=training_cfg["chunk_size"],
        dim_model=training_cfg["dim_model"],
        n_heads=training_cfg["n_heads"],
        n_encoder_layers=training_cfg["n_encoder_layers"],
        optimizer_lr=training_cfg["learning_rate"],
    )

    # Create policy and processors
    policy = ACTPolicy(policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset_metadata.stats,
    )

    policy.train()
    policy.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    typer.echo(f"  Total parameters: {total_params:,}")
    typer.echo(f"  Trainable parameters: {trainable_params:,}")

    # Setup delta timestamps for action chunking
    delta_timestamps = {
        "action": make_delta_timestamps(policy_cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(policy_cfg.observation_delta_indices, dataset_metadata.fps)
        for k in policy_cfg.image_features
    }

    # Load dataset
    typer.echo("\nLoading full dataset...")
    video_backend = training_cfg.get("video_backend")
    dataset = LeRobotDataset(
        dataset_repo_id, root=local_path, delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    typer.echo(f"  Episodes: {dataset.num_episodes}")
    typer.echo(f"  Frames: {len(dataset)}")

    # Create optimizer and dataloader
    optimizer = policy_cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=4,
    )

    # Setup output directory: CLI flag > config > default (data/outputs/)
    if output_dir_override:
        output_dir = Path(output_dir_override).expanduser()
    elif training_cfg["output_dir"]:
        output_dir = Path(training_cfg["output_dir"]).expanduser()
    else:
        output_dir = OUTPUTS_DIR / training_cfg["output_repo_id"].replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if wandb_enabled:
        typer.echo("\nInitializing Weights & Biases...")
        wandb_config = {
            "dataset": training_cfg["dataset_repo_id"],
            "policy": training_cfg["policy"],
            "steps": training_cfg["steps"],
            "batch_size": training_cfg["batch_size"],
            "learning_rate": training_cfg["learning_rate"],
            "chunk_size": training_cfg["chunk_size"],
            "dim_model": training_cfg["dim_model"],
            "n_heads": training_cfg["n_heads"],
            "n_encoder_layers": training_cfg["n_encoder_layers"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dataset_episodes": dataset.num_episodes,
            "dataset_frames": len(dataset),
        }

        if resume_run_id:
            wandb.init(
                project=training_cfg["wandb_project"],
                entity=training_cfg["wandb_entity"],
                id=resume_run_id,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(
                project=training_cfg["wandb_project"],
                entity=training_cfg["wandb_entity"],
                config=wandb_config,
                name=f"act-{training_cfg['dataset_repo_id'].split('/')[-1]}",
            )

    # Training loop
    typer.echo("\n=== Starting Training ===\n")
    training_steps = steps_override or training_cfg["steps"]
    log_freq = training_cfg["log_freq"]
    save_freq = training_cfg["save_freq"]

    step = 0
    epoch = 0
    done = False
    start_time = time.time()

    try:
        while not done:
            epoch += 1
            for raw_batch in dataloader:
                batch = preprocessor(raw_batch)
                loss, loss_dict = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1

                # Logging
                if step % log_freq == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed
                    eta_seconds = (
                        (training_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    )
                    eta_hours = eta_seconds / 3600

                    mem_str = ""
                    if device.type == "cuda":
                        mem_gb = torch.cuda.memory_allocated(device) / 1e9
                        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                        mem_str = f" | VRAM: {mem_gb:.1f}GB cur, {peak_gb:.1f}GB peak, {total_gb:.0f}GB total"

                    typer.echo(
                        f"Step {step}/{training_steps} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Speed: {steps_per_sec:.1f} steps/s | "
                        f"ETA: {eta_hours:.1f}h{mem_str}"
                    )

                    if wandb_enabled:
                        log_data = {
                            "train/loss": loss.item(),
                            "train/step": step,
                            "train/epoch": epoch,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        if device.type == "cuda":
                            log_data["train/vram_gb"] = mem_gb
                            log_data["train/vram_peak_gb"] = peak_gb
                            log_data["train/vram_total_gb"] = total_gb
                        # Add individual loss components
                        for key, value in loss_dict.items():
                            log_data[f"train/{key}"] = (
                                value.item() if hasattr(value, "item") else value
                            )
                        wandb.log(log_data, step=step)

                # Save checkpoint
                if step % save_freq == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    typer.echo(f"\nSaving checkpoint at step {step}...")
                    policy.save_pretrained(checkpoint_dir)
                    preprocessor.save_pretrained(checkpoint_dir)
                    postprocessor.save_pretrained(checkpoint_dir)

                    if wandb_enabled:
                        wandb.log({"train/checkpoint_saved": step}, step=step)

                if step >= training_steps:
                    done = True
                    break

    except KeyboardInterrupt:
        typer.echo("\n\nTraining interrupted by user.")
        if yes or typer.confirm("Save current checkpoint before exiting?"):
            checkpoint_dir = output_dir / f"checkpoint-{step}-interrupted"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)
            typer.echo(f"Checkpoint saved to {checkpoint_dir}")

    # Save final model
    typer.echo("\n=== Training Complete ===\n")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Saving final model...")
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    typer.echo(f"Model saved to {final_dir}")

    # Push to HuggingFace Hub
    if push_to_hub:
        typer.echo(f"\nPushing model to HuggingFace Hub: {training_cfg['output_repo_id']}")
        policy.push_to_hub(training_cfg["output_repo_id"])
        preprocessor.push_to_hub(training_cfg["output_repo_id"])
        postprocessor.push_to_hub(training_cfg["output_repo_id"])
        typer.echo("Model pushed successfully!")

        if wandb_enabled:
            wandb.log({"hub/model_pushed": True}, step=step)

    # Finish wandb
    if wandb_enabled:
        wandb.finish()

    total_time = time.time() - start_time
    typer.echo(f"\nTotal training time: {total_time / 3600:.2f} hours")
    if step > 0:
        typer.echo(f"Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    app()
