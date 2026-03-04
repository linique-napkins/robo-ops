# Training

Two independent training pipelines for bimanual SO101 robot:

```
training/
├── act/     ACT action-chunking policy (predicts motor commands, runs on robot)
└── sarm/    SARM reward model (predicts task progress 0→1, computes RA-BC weights)
```

```
┌─────────────────────────────────┐     ┌──────────────────────────────────────┐
│  ACT Pipeline                    │     │  SARM + RA-BC Pipeline                │
│                                  │     │                                       │
│  uv run training/act/train.py    │     │  1. uv run training/sarm/train.py     │
│          │                       │     │          │                            │
│          ▼                       │     │  2. Inspect predictions               │
│  Run on robot                    │     │          │                            │
│  uv run inference/run.py         │     │  3. Compute RA-BC weights             │
│                                  │     │          │                            │
│                                  │     │  4. lerobot-train --policy.type=pi0   │
│                                  │     │     --use_rabc=true                   │
│                                  │     │          │                            │
│                                  │     │  5. Run on robot                      │
│                                  │     │     uv run inference/run.py            │
└─────────────────────────────────┘     └──────────────────────────────────────┘
```

See [act/README.md](act/README.md) and [sarm/README.md](sarm/README.md) for pipeline-specific docs.

---

## Resuming Cancelled Training

Both pipelines save checkpoints periodically (every `save_freq` steps) and on Ctrl+C interruption. To resume:

```bash
# ACT — resume from a checkpoint (+ optionally resume the wandb run)
uv run training/act/train.py --checkpoint data/outputs/.../checkpoint-10000 \
                              --resume <wandb_run_id>

# SARM — same pattern
uv run training/sarm/train.py --checkpoint data/outputs/.../checkpoint-1000 \
                               --resume <wandb_run_id>
```

`--checkpoint` restores model weights, optimizer state, and step counter. `--resume` continues logging to an existing wandb run (find the run ID on the wandb dashboard). Both flags are independent — you can use either or both.

Checkpoint directory contents:

```
checkpoint-10000/
├── model.safetensors       # Policy/model weights
├── config.json             # Model config
├── preprocessor_config.json
├── postprocessor_config.json
├── optimizer.pt            # Optimizer state
├── scheduler.pt            # LR scheduler state (SARM only)
└── training_state.pt       # Step counter
```

If `training_state.pt` is missing (e.g. from an older checkpoint), the step number is inferred from the directory name.

---

## Sockeye GPU Hardware

| Spec | Details |
|------|---------|
| GPU model | NVIDIA Tesla V100-SXM2 |
| GPU memory | **32 GB** (44 nodes) or 16 GB (6 nodes) |
| GPUs per node | 4 |
| CPU cores per node | 24 (= 6 per GPU) |
| RAM per node | 192 GB |
| Total GPUs | 200 across 50 nodes |

## Storage Tiers

| Tier | Path | Quota | Purge | Compute Access |
|------|------|-------|-------|----------------|
| Home | `/home/$USER` | 50 GB | No | **READ-ONLY** |
| Project | `/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding` | 5 TB | No | **READ-ONLY** |
| Scratch | `/scratch/ss-engineeringphysics-1/$USER` | 5 TB | **Yes** | Read/Write |

- **Home and Project are read-only on compute nodes.** Compute nodes also have **no internet access**. All output must go to scratch, and any downloads (model weights, packages) must be done from the login node beforehand.
- **Scratch gets purged.** Copy results to project storage after training.
