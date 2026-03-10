# SARM Reward Model Training

Stage-Aware Reward Model (SARM) for bimanual SO101 robot. SARM predicts task progress (0→1) from video — it does NOT produce actions or run on the robot. Its output is used to compute RA-BC weights for downstream VLA policy training (PI0, SmolVLA) via lerobot's `lerobot-train`.

SARM uses CLIP (`openai/clip-vit-base-patch32`, ~150MB, frozen) to encode video frames into 512-dim vectors, then trains transformer heads to classify stages and predict within-stage progress (tau).

## Quick Start

```bash
uv run training/sarm/train.py                            # full training
uv run training/sarm/train.py --steps 5 --no-wandb --yes  # smoke test
```

## RA-BC Workflow

```bash
# 1. Train SARM reward model
uv run training/sarm/train.py

# 2. Inspect SARM predictions (visualize progress curves)
uv run lerobot/src/lerobot/policies/sarm/compute_rabc_weights.py \
  --dataset-repo-id jhimmens/linique-v2 \
  --reward-model-path data/outputs/jhimmens_linique-sarm/final \
  --head-mode sparse \
  --visualize-only \
  --num-visualizations 5

# 3. Compute RA-BC weights
uv run lerobot/src/lerobot/policies/sarm/compute_rabc_weights.py \
  --dataset-repo-id jhimmens/linique-v2 \
  --reward-model-path data/outputs/jhimmens_linique-sarm/final \
  --head-mode sparse

# 4. Train VLA with RA-BC weighting
lerobot-train \
  --dataset.repo_id=jhimmens/linique-v2 \
  --policy.type=pi0 \
  --use_rabc=true \
  --rabc_head_mode=sparse \
  --rabc_kappa=0.01 \
  --wandb.enable=true \
  --wandb.project=linique-robot

# 5. Run on robot (update inference/config.toml with model path)
uv run inference/run.py
```

## Annotation Modes

Configured in `config.toml`:

**`single_stage`** (default) — no annotations needed. Treats the entire episode as one stage and learns overall task completion progress (0→1). Good starting point.

**`dense_only`** — fine-grained subtask annotations. SARM learns per-subtask progress. Requires annotating the dataset first:

```bash
uv run lerobot/src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
  --repo-id jhimmens/linique-v2 \
  --dense-only \
  --dense-subtasks "Bring robot arms up,Grab near side for 1st fold,Do 2nd fold,Do 3rd fold" \
  --video-key observation.images.left_top_cam
```

Then set in `config.toml`:

```toml
annotation_mode = "dense_only"
dense_subtask_names = ["Bring robot arms up", "Grab near side for 1st fold", ...]
dense_temporal_proportions = [0.15, 0.35, 0.25, 0.25]  # must sum to 1.0
```

When computing RA-BC weights after dense training, use `--head-mode dense`.

## Inspecting Predictions

After training, inspect predictions before computing RA-BC weights:

```bash
uv run lerobot/src/lerobot/policies/sarm/compute_rabc_weights.py \
  --dataset-repo-id jhimmens/linique-v2 \
  --reward-model-path data/outputs/jhimmens_linique-sarm/final \
  --head-mode sparse \
  --visualize-only \
  --num-visualizations 5
```

Each visualization PNG has three panels:
- **Progress curve** — predicted progress (0→1) over frame index. Should be roughly monotonic.
- **Stage probability stackplot** — confidence across stages over time. Should be decisive.
- **Frame strip** — 8 sampled frames with predicted progress and stage labels.

## Configuration

`config.toml` uses environment-aware sections:

```toml
[sarm]
steps = 5000
hidden_dim = 768
# ...

[sarm.env.local]
batch_size = 8

[sarm.env.sockeye]
batch_size = 32
```

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
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/training_outputs/sarm/ \
    data/outputs/jhimmens_linique-sarm-sockeye/

# Pull wandb offline runs and sync
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/robo-ops/wandb/offline-run-* \
    wandb/
uv run wandb sync wandb/offline-run-*
```

### On the Sockeye login node

```bash
cd /scratch/ss-engineeringphysics-1/$USER/robo-ops
source $HOME/venvs/robo-ops/bin/activate
uv sync --active

# Pre-download CLIP weights (compute nodes have no internet)
uv run training/sarm/download_clip.py
```

### Submit

```bash
sbatch --test-only training/sarm/arc_train.sh              # dry run
TRAIN_STEPS=5 sbatch --time=0:15:00 training/sarm/arc_train.sh  # quick test
sbatch training/sarm/arc_train.sh                          # full training
```

### Monitor

```bash
squeue -u $USER
tail -f /scratch/ss-engineeringphysics-1/$USER/training_outputs/sarm-output-<jobid>.txt

# After training: copy to project storage (scratch gets purged!)
cp -r /scratch/ss-engineeringphysics-1/$USER/training_outputs/sarm \
    /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/models/
```

## Troubleshooting

**OOM** — SARM is lighter than ACT (CLIP encoder is frozen). `batch_size = 32` should fit on V100-32GB. If OOM, reduce `batch_size` in `config.toml` under `[sarm.env.sockeye]`.

**wandb fails to sync** — Compute nodes have no internet. `arc_train.sh` uses `WANDB_MODE=offline`. Sync from login node: `wandb sync wandb/offline-run-*`.
