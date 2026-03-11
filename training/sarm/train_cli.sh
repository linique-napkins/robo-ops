#!/bin/bash
# Train SARM using the lerobot CLI, reading settings from config.toml
#
# Usage:
#   bash training/sarm/train_cli.sh              # use config.toml defaults
#   bash training/sarm/train_cli.sh --steps 5    # override steps
#   DRY_RUN=1 bash training/sarm/train_cli.sh    # print command without running

set -euo pipefail

CONFIG="training/sarm/config.toml"

# ── Parse config.toml ──────────────────────────────────────
cfg() { grep "^$1 " "$CONFIG" | sed 's/.*= *"\{0,1\}\([^"]*\)"\{0,1\}/\1/' | tr -d ' '; }

DATASET_REPO_ID=$(cfg dataset_repo_id)
OUTPUT_REPO_ID=$(cfg output_repo_id)
ANNOTATION_MODE=$(cfg annotation_mode)
IMAGE_KEY=$(cfg image_key)
STATE_KEY=$(cfg state_key)
HIDDEN_DIM=$(cfg hidden_dim)
NUM_HEADS=$(cfg num_heads)
NUM_LAYERS=$(cfg num_layers)
N_OBS_STEPS=$(cfg n_obs_steps)
FRAME_GAP=$(cfg frame_gap)
MAX_REWIND=$(cfg max_rewind_steps)
STEPS=$(cfg steps)
DEVICE=$(cfg device)
SAVE_FREQ=$(cfg save_freq)
LOG_FREQ=$(cfg log_freq)
WANDB_PROJECT=$(cfg wandb_project)
VIDEO_BACKEND=$(cfg video_backend)

# Batch size: check env-specific first, fall back to top-level
BATCH_SIZE=$(cfg batch_size)

# ── Build command ──────────────────────────────────────────
CMD=(
  uv run lerobot-train
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="data/datasets/$DATASET_REPO_ID"
  --dataset.video_backend="$VIDEO_BACKEND"
  --policy.type=sarm
  --policy.annotation_mode="$ANNOTATION_MODE"
  --policy.image_key="$IMAGE_KEY"
  --policy.state_key="$STATE_KEY"
  --policy.hidden_dim="$HIDDEN_DIM"
  --policy.num_heads="$NUM_HEADS"
  --policy.num_layers="$NUM_LAYERS"
  --policy.n_obs_steps="$N_OBS_STEPS"
  --policy.frame_gap="$FRAME_GAP"
  --policy.max_rewind_steps="$MAX_REWIND"
  --policy.repo_id="$OUTPUT_REPO_ID"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --log_freq="$LOG_FREQ"
  --save_freq="$SAVE_FREQ"
  --device="$DEVICE"
  --wandb.enable=true
  --wandb.project="$WANDB_PROJECT"
)

# Append any extra args passed to this script (e.g. --steps 5)
CMD+=("$@")

echo "=== SARM Training (lerobot CLI) ==="
echo "${CMD[*]}"
echo

if [ "${DRY_RUN:-}" = "1" ]; then
  echo "(dry run, not executing)"
  exit 0
fi

exec "${CMD[@]}"
