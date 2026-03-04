"""
Training script for SARM (Stage-Aware Reward Model) on bimanual SO101 robot data.

SARM learns task progress prediction (0→1) from video demonstrations. Unlike ACT which
predicts actions, SARM learns *how well* a task is progressing. The trained model is then
used to compute RA-BC weights for ACT training.

Uses wandb for experiment tracking and pushes trained models to HuggingFace Hub.

Usage:
    uv run training/sarm/train.py
    uv run training/sarm/train.py --steps 5 --no-wandb --yes  # smoke test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import UTC
from datetime import datetime

import torch
import typer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

import wandb
from lib.config import OUTPUTS_DIR
from lib.config import get_local_dataset_path
from lib.config import load_training_config

app = typer.Typer()

LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"

# Gradient clipping norm (matches AdamWConfig default in lerobot)
GRAD_CLIP_NORM = 10.0


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


def print_training_config(sarm_cfg: dict) -> None:
    """Print SARM training configuration summary."""
    typer.echo("\nSARM Training Configuration:")
    typer.echo(f"  Dataset:          {sarm_cfg['dataset_repo_id']}")
    typer.echo(f"  Output repo:      {sarm_cfg['output_repo_id']}")
    typer.echo(f"  Annotation mode:  {sarm_cfg['annotation_mode']}")
    typer.echo(f"  Image key:        {sarm_cfg['image_key']}")
    typer.echo(f"  State key:        {sarm_cfg['state_key']}")
    typer.echo(f"  Steps:            {sarm_cfg['steps']}")
    typer.echo(f"  Batch size:       {sarm_cfg['batch_size']}")
    typer.echo(f"  Device:           {sarm_cfg['device']}")
    typer.echo(f"  Hidden dim:       {sarm_cfg['hidden_dim']}")
    typer.echo(f"  Num layers:       {sarm_cfg['num_layers']}")
    typer.echo(f"  n_obs_steps:      {sarm_cfg['n_obs_steps']}")
    typer.echo(f"  frame_gap:        {sarm_cfg['frame_gap']}")
    typer.echo(f"  Output dir:       {sarm_cfg.get('output_dir') or 'default (data/outputs/)'}")
    typer.echo(f"  Save freq:        every {sarm_cfg['save_freq']} steps")
    typer.echo(f"  Log freq:         every {sarm_cfg['log_freq']} steps")

    if sarm_cfg["annotation_mode"] == "dense_only" and sarm_cfg.get("dense_subtask_names"):
        typer.echo(f"\n  Dense subtasks ({len(sarm_cfg['dense_subtask_names'])}):")
        for name, prop in zip(
            sarm_cfg["dense_subtask_names"],
            sarm_cfg["dense_temporal_proportions"],
            strict=True,
        ):
            typer.echo(f"    - {name}  ({prop:.0%} of episode)")


def validate_dense_config(cfg: dict) -> None:
    """Validate dense subtask config when annotation_mode = 'dense_only'."""
    if cfg.get("annotation_mode") != "dense_only":
        return
    if not cfg.get("dense_subtask_names"):
        msg = (
            "dense_subtask_names is required when annotation_mode = 'dense_only'. "
            "Add a list of subtask names to [sarm] in config.toml."
        )
        raise ValueError(msg)
    if not cfg.get("dense_temporal_proportions"):
        msg = (
            "dense_temporal_proportions is required when annotation_mode = 'dense_only'. "
            "Add a list of floats (summing to 1.0) to [sarm] in config.toml."
        )
        raise ValueError(msg)
    if len(cfg["dense_subtask_names"]) != len(cfg["dense_temporal_proportions"]):
        msg = (
            f"dense_subtask_names ({len(cfg['dense_subtask_names'])}) and "
            f"dense_temporal_proportions ({len(cfg['dense_temporal_proportions'])}) "
            "must have the same length."
        )
        raise ValueError(msg)
    prop_sum = sum(cfg["dense_temporal_proportions"])
    if abs(prop_sum - 1.0) > 0.01:
        msg = f"dense_temporal_proportions must sum to 1.0, got {prop_sum:.4f}"
        raise ValueError(msg)


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
    checkpoint_path: str = typer.Option(
        None,
        "--checkpoint",
        help="Path to checkpoint dir to resume from (e.g. data/outputs/.../checkpoint-1000)",
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
    """Train SARM reward model on recorded robot data."""
    typer.echo("\n=== SARM Reward Model Training ===\n")

    # Load training configuration
    sarm_cfg = load_training_config(LOCAL_CONFIG_PATH, "sarm")
    validate_dense_config(sarm_cfg)

    print_training_config(sarm_cfg)

    if not yes and not typer.confirm("\nProceed with training?"):
        typer.echo("Training cancelled.")
        raise typer.Exit(0)

    # Setup device
    device = get_device(sarm_cfg["device"])
    typer.echo(f"\nUsing device: {device}")

    # Load dataset metadata and features
    dataset_repo_id = sarm_cfg["dataset_repo_id"]
    local_path = get_local_dataset_path(dataset_repo_id)
    typer.echo(f"\nLoading dataset: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=local_path)

    # Create SARM configuration
    typer.echo("\nCreating SARM reward model...")
    sarm_kwargs: dict = {
        "annotation_mode": sarm_cfg["annotation_mode"],
        "image_key": sarm_cfg["image_key"],
        "state_key": sarm_cfg["state_key"],
        "hidden_dim": sarm_cfg["hidden_dim"],
        "num_heads": sarm_cfg["num_heads"],
        "num_layers": sarm_cfg["num_layers"],
        "n_obs_steps": sarm_cfg["n_obs_steps"],
        "frame_gap": sarm_cfg["frame_gap"],
        "max_rewind_steps": sarm_cfg["max_rewind_steps"],
        "batch_size": sarm_cfg["batch_size"],
        "device": str(device),
    }

    # Pass dense subtask config when using dense_only mode
    if sarm_cfg["annotation_mode"] == "dense_only":
        sarm_kwargs["dense_subtask_names"] = sarm_cfg["dense_subtask_names"]
        sarm_kwargs["dense_temporal_proportions"] = sarm_cfg["dense_temporal_proportions"]
        sarm_kwargs["num_dense_stages"] = len(sarm_cfg["dense_subtask_names"])

    policy_cfg = SARMConfig(**sarm_kwargs)

    # Create model (or load from checkpoint)
    resume_step = 0
    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path)
        typer.echo(f"\nLoading checkpoint from {ckpt_dir}...")
        model = SARMRewardModel.from_pretrained(ckpt_dir)
        state_path = ckpt_dir / "training_state.pt"
        if state_path.exists():
            training_state = torch.load(state_path, map_location=device, weights_only=True)
            resume_step = training_state["step"]
            typer.echo(f"  Resuming from step {resume_step}")
        else:
            ckpt_name = ckpt_dir.name
            for part in ckpt_name.split("-"):
                if part.isdigit():
                    resume_step = int(part)
                    break
            typer.echo(f"  No training_state.pt, inferred step {resume_step} from dir name")
    else:
        model = SARMRewardModel(policy_cfg)
    model.train()
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"  Total parameters: {total_params:,}")
    typer.echo(f"  Trainable parameters: {trainable_params:,}")

    # Setup delta timestamps — SARM uses observation_delta_indices only (no actions)
    delta_timestamps = {
        sarm_cfg["image_key"]: make_delta_timestamps(
            policy_cfg.observation_delta_indices, dataset_metadata.fps
        ),
        sarm_cfg["state_key"]: make_delta_timestamps(
            policy_cfg.observation_delta_indices, dataset_metadata.fps
        ),
    }

    # Load dataset
    typer.echo("\nLoading full dataset...")
    video_backend = sarm_cfg.get("video_backend")
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=local_path,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    typer.echo(f"  Episodes: {dataset.num_episodes}")
    typer.echo(f"  Frames: {len(dataset)}")

    # Create preprocessor with dataset_meta for episode-aware processing
    preprocessor, postprocessor = make_sarm_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    # Compute training steps early (scheduler depends on it)
    training_steps = steps_override or sarm_cfg["steps"]

    # Create optimizer, LR scheduler, and dataloader
    # SARM uses EpisodeAwareSampler to drop last frames per episode
    optimizer = policy_cfg.get_optimizer_preset().build(model.parameters())
    lr_scheduler = policy_cfg.get_scheduler_preset().build(optimizer, training_steps)

    # Restore optimizer and scheduler state if resuming from checkpoint
    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path)
        optimizer_path = ckpt_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=device, weights_only=True)
            )
            typer.echo("  Optimizer state restored")
        else:
            typer.echo("  Warning: no optimizer.pt found, optimizer starting fresh")
        scheduler_path = ckpt_dir / "scheduler.pt"
        if scheduler_path.exists():
            lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=device, weights_only=True)
            )
            typer.echo("  LR scheduler state restored")
        else:
            typer.echo("  Warning: no scheduler.pt found, scheduler starting fresh")
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=policy_cfg.drop_n_last_frames,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sarm_cfg["batch_size"],
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=4,
    )

    # Setup base output directory: CLI flag > config > default (data/outputs/)
    if output_dir_override:
        base_output_dir = Path(output_dir_override).expanduser()
    elif sarm_cfg.get("output_dir"):
        base_output_dir = Path(sarm_cfg["output_dir"]).expanduser()
    else:
        base_output_dir = OUTPUTS_DIR / sarm_cfg["output_repo_id"].replace("/", "_")

    # Initialize wandb
    if wandb_enabled:
        typer.echo("\nInitializing Weights & Biases...")
        wandb_config = {
            "model": "sarm",
            "dataset": sarm_cfg["dataset_repo_id"],
            "annotation_mode": sarm_cfg["annotation_mode"],
            "steps": sarm_cfg["steps"],
            "batch_size": sarm_cfg["batch_size"],
            "hidden_dim": sarm_cfg["hidden_dim"],
            "num_heads": sarm_cfg["num_heads"],
            "num_layers": sarm_cfg["num_layers"],
            "n_obs_steps": sarm_cfg["n_obs_steps"],
            "frame_gap": sarm_cfg["frame_gap"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dataset_episodes": dataset.num_episodes,
            "dataset_frames": len(dataset),
        }
        if sarm_cfg.get("dense_subtask_names"):
            wandb_config["dense_subtask_names"] = sarm_cfg["dense_subtask_names"]
            wandb_config["num_dense_stages"] = len(sarm_cfg["dense_subtask_names"])

        if resume_run_id:
            wandb.init(
                project=sarm_cfg["wandb_project"],
                entity=sarm_cfg.get("wandb_entity"),
                id=resume_run_id,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(
                project=sarm_cfg["wandb_project"],
                entity=sarm_cfg.get("wandb_entity"),
                config=wandb_config,
            )

    # Create run-specific output directory using wandb run name or datetime
    if wandb_enabled and wandb.run is not None:
        run_name = wandb.run.name
    else:
        run_name = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"  Output directory: {output_dir}")

    # Training loop
    typer.echo("\n=== Starting Training ===\n")
    log_freq = sarm_cfg["log_freq"]
    save_freq = sarm_cfg["save_freq"]

    step = resume_step
    epoch = 0
    done = False
    start_time = time.time()

    try:
        while not done:
            epoch += 1
            for raw_batch in dataloader:
                batch = preprocessor(raw_batch)
                loss, loss_dict = model.forward(batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                lr_scheduler.step()
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
                        mem_str = (
                            f" | VRAM: {mem_gb:.1f}GB cur,"
                            f" {peak_gb:.1f}GB peak, {total_gb:.0f}GB total"
                        )

                    # Build loss component string
                    loss_parts = []
                    for key, value in loss_dict.items():
                        v = value.item() if hasattr(value, "item") else value
                        loss_parts.append(f"{key}: {v:.4f}")
                    loss_str = " | ".join(loss_parts) if loss_parts else ""

                    typer.echo(
                        f"Step {step}/{training_steps} | "
                        f"Loss: {loss.item():.4f} | "
                        f"{loss_str} | "
                        f"Speed: {steps_per_sec:.1f} steps/s | "
                        f"ETA: {eta_hours:.1f}h{mem_str}"
                    )

                    if wandb_enabled:
                        log_data = {
                            "train/loss": loss.item(),
                            "train/step": step,
                            "train/epoch": epoch,
                            "train/steps_per_sec": steps_per_sec,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/grad_norm": grad_norm.item(),
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
                    model.save_pretrained(checkpoint_dir)
                    preprocessor.save_pretrained(checkpoint_dir)
                    postprocessor.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                    torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
                    torch.save({"step": step}, checkpoint_dir / "training_state.pt")

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
            model.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
            torch.save({"step": step}, checkpoint_dir / "training_state.pt")
            typer.echo(f"Checkpoint saved to {checkpoint_dir}")

    # Save final model
    typer.echo("\n=== Training Complete ===\n")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Saving final model...")
    model.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    typer.echo(f"Model saved to {final_dir}")

    # Push to HuggingFace Hub
    if push_to_hub:
        typer.echo(f"\nPushing model to HuggingFace Hub: {sarm_cfg['output_repo_id']}")
        model.push_to_hub(sarm_cfg["output_repo_id"])
        preprocessor.push_to_hub(sarm_cfg["output_repo_id"])
        postprocessor.push_to_hub(sarm_cfg["output_repo_id"])
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
