# ACT Policy Training

Action Chunking with Transformers (ACT) policy for bimanual SO101 robot.

## Quick Start

```bash
uv run training/act/train.py                            # full training
uv run training/act/train.py --steps 5 --no-wandb --yes  # smoke test
```

## Configuration

`config.toml` uses environment-aware sections. Shared values are at the top level, environment-specific values (batch size, learning rate) are in `[training.env.local]` and `[training.env.sockeye]`:

```toml
[training]
steps = 100000
chunk_size = 100
dim_model = 512
# ...

[training.env.local]
batch_size = 8
learning_rate = 1e-5

[training.env.sockeye]
batch_size = 38
learning_rate = 4.75e-5
```

The environment is auto-detected from `$USER` (`nvd` = local, `jhimmens` = sockeye).

### Batch Size Guidance for V100-32GB

With ACT (dim_model=512, chunk_size=100, 3 cameras at 640x480):
- `batch_size = 8`: ~12 GB VRAM, safe starting point
- `batch_size = 16`: ~20 GB VRAM, good throughput
- `batch_size = 32`: ~28 GB VRAM, near the limit

## Sockeye

### On your server (has internet)

```bash
# Sync dataset to Sockeye project storage
rsync -avzP data/datasets/jhimmens/linique-v2 \
    jhimmens@dtn.sockeye.arc.ubc.ca:/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/datasets/

# Sync repo to scratch
rsync -avzP --exclude data/ --exclude .venv/ --exclude wandb/ \
    . jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/robo-ops/

# Pull training outputs back after training
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/training_outputs/ \
    data/outputs/jhimmens_linique-act-v2-sockeye/

# Pull wandb offline runs and sync
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/robo-ops/wandb/offline-run-* \
    wandb/
uv run wandb sync wandb/offline-run-*
```

### On the Sockeye login node

```bash
cd /scratch/ss-engineeringphysics-1/$USER/robo-ops

# Pre-download ResNet weights (compute nodes have no internet)
mkdir -p /scratch/ss-engineeringphysics-1/$USER/.cache/torch/hub/checkpoints
wget -P /scratch/ss-engineeringphysics-1/$USER/.cache/torch/hub/checkpoints \
    https://download.pytorch.org/models/resnet18-f37072fd.pth
```

### Submit

```bash
sbatch --test-only training/act/arc_train.sh              # dry run
TRAIN_STEPS=5 sbatch --time=0:15:00 training/act/arc_train.sh  # quick test
sbatch training/act/arc_train.sh                          # full training
```

### Monitor

```bash
squeue -u $USER
tail -f /scratch/ss-engineeringphysics-1/$USER/training_outputs/output-<jobid>.txt

# After training: copy to project storage (scratch gets purged!)
cp -r /scratch/ss-engineeringphysics-1/$USER/training_outputs \
    /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/models/
```

## Troubleshooting

**OOM** — Reduce `batch_size` in `config.toml`, enable AMP, or check for 16 GB nodes (`--constraint=gpu_mem_32`).

**wandb fails to sync** — Compute nodes have no internet. `arc_train.sh` uses `WANDB_MODE=offline`. Sync from login node: `wandb sync wandb/offline-run-*`.
