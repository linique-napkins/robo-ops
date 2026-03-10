# Pi0 Policy Training

Pi0 (Physical Intelligence) vision-language-action policy for bimanual SO101 robot. Fine-tunes a pretrained Pi0 base model (`lerobot/pi0_base`) on your demonstration dataset using `lerobot-train`.

Pi0 uses PaliGemma (vision-language encoder) + Gemma (action expert) with flow matching for action generation. Much larger than ACT (~3B params vs ~10M) but can leverage vision-language pretraining.

## Quick Start

```bash
# Fine-tune Pi0 on your dataset
lerobot-train \
  --policy.path=lerobot/pi0_base \
  --dataset.repo_id=jhimmens/linique-v2 \
  --dataset.root=data/datasets \
  --wandb.enable=true \
  --wandb.project=linique-robot

# With RA-BC weighting from SARM (see sarm/README.md for full workflow)
lerobot-train \
  --policy.path=lerobot/pi0_base \
  --dataset.repo_id=jhimmens/linique-v2 \
  --dataset.root=data/datasets \
  --use_rabc=true \
  --rabc_head_mode=sparse \
  --rabc_kappa=0.01 \
  --wandb.enable=true \
  --wandb.project=linique-robot
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
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/training_outputs/pi0/ \
    data/outputs/jhimmens_linique-pi0-sockeye/

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

# Pre-download Pi0 base weights + PaliGemma tokenizer (compute nodes have no internet)
uv run training/pi0/download_weights.py
```

### Submit

```bash
sbatch --test-only training/pi0/arc_train.sh              # dry run
TRAIN_STEPS=5 sbatch --time=0:15:00 training/pi0/arc_train.sh  # quick test
sbatch training/pi0/arc_train.sh                          # full training

# With RA-BC weighting (requires SARM training + weight computation first)
USE_RABC=1 sbatch training/pi0/arc_train.sh
```

### Monitor

```bash
squeue -u $USER
tail -f /scratch/ss-engineeringphysics-1/$USER/training_outputs/pi0-output-<jobid>.txt

# After training: copy to project storage (scratch gets purged!)
cp -r /scratch/ss-engineeringphysics-1/$USER/training_outputs/pi0 \
    /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/models/
```

## Troubleshooting

**OOM** — Pi0 is much larger than ACT (~3B params). On V100-32GB, start with small batch sizes. Use `--policy.grad_checkpoint=true` to trade compute for memory. If still OOM, consider LoRA fine-tuning with `--peft.type=lora`.

**wandb fails to sync** — Compute nodes have no internet. `arc_train.sh` uses `WANDB_MODE=offline`. Sync from login node: `wandb sync wandb/offline-run-*`.

**Slow training** — Pi0 is ~300x larger than ACT. Expect slower steps. Use `--policy.use_amp=true` for mixed precision on V100.
