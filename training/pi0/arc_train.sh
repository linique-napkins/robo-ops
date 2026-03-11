#!/bin/bash

#SBATCH --job-name=ENPH-2617-Linique-PI0
#SBATCH --account=ss-engineeringphysics-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=gpu_mem_32
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/ss-engineeringphysics-1/%u/training_outputs/pi0-output-%j.txt
#SBATCH --error=/scratch/ss-engineeringphysics-1/%u/training_outputs/pi0-error-%j.txt
#SBATCH --mail-user=jhimmens@student.ubc.ca
#SBATCH --mail-type=ALL

# ── Paths ───────────────────────────────────────────────────
ALLOC=ss-engineeringphysics-1
SCRATCH=/scratch/$ALLOC/$USER
PROJECT=/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding
OUTPUT_BASE=$SCRATCH/training_outputs/pi0
OUTPUT_DIR=$OUTPUT_BASE/$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOB_ID}

# ── Environment ─────────────────────────────────────────────
module load gcc/9.4.0
module load cuda
module load ffmpeg/6.0
export LD_LIBRARY_PATH=$FFMPEG_ROOT/lib:$LD_LIBRARY_PATH

source $HOME/venvs/robo-ops/bin/activate

REPO_DIR=$SCRATCH/robo-ops
cd $REPO_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export WANDB_DIR=$OUTPUT_DIR
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb

# ── Pre-download Pi0 weights (cached in HF_HOME) ───────────
# Pi0 uses PaliGemma + Gemma. Pre-download on login node if needed:
#   uv run training/pi0/download_weights.py
# Weights are cached at $HF_HOME/hub/models--lerobot--pi0_base/

# ── Sync dataset from project storage (read-only) ──────────
DATASET_REPO_ID=${DATASET_REPO_ID:-"jhimmens/linique-v2"}
DATASET_NAME=${DATASET_REPO_ID##*/}   # e.g. "linique-v2"
DATASET_OWNER=${DATASET_REPO_ID%%/*}  # e.g. "jhimmens"

mkdir -p "data/datasets/$DATASET_OWNER"
ln -sfn "$PROJECT/datasets/$DATASET_NAME" "data/datasets/$DATASET_REPO_ID"

# ── Output base on scratch (writable) ─────────────────────
mkdir -p $OUTPUT_BASE

# ── GPU info ──────────────────────────────────────────────────
nvidia-smi --query-gpu=uuid --format=csv,noheader

# ── Train Pi0 ──────────────────────────────────────────────
TRAIN_ARGS=(
    --policy.path=lerobot/pi0_base
    --dataset.repo_id=$DATASET_REPO_ID
    --dataset.root=data/datasets/$DATASET_REPO_ID
    --dataset.video_backend=pyav
    --output_dir=$OUTPUT_DIR
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=true
    --policy.freeze_vision_encoder=true
    --batch_size=1
    --save_freq=1000
    --wandb.enable=true
    --wandb.project=linique-pi0
    --wandb.mode=offline
    --policy.use_amp=true
    --peft.method_type=LORA
    --peft.r=16
    --rename_map='{"observation.images.left_top_cam": "observation.images.base_0_rgb", "observation.images.left_left_cam": "observation.images.left_wrist_0_rgb", "observation.images.left_right_cam": "observation.images.right_wrist_0_rgb"}'
)

# Optional: RA-BC weighting from SARM reward model
if [ -n "$USE_RABC" ]; then
    TRAIN_ARGS+=(
        --use_rabc=true
        --rabc_head_mode=${RABC_HEAD_MODE:-sparse}
        --rabc_kappa=${RABC_KAPPA:-0.01}
    )
fi

# Optional: override training steps
if [ -n "$TRAIN_STEPS" ]; then
    TRAIN_ARGS+=(--steps=$TRAIN_STEPS)
fi

# Multi-GPU FSDP (commented out — use LoRA instead for OOM)
# accelerate launch \
#     --num_processes=2 \
#     --use_fsdp \
#     --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
#     --fsdp_sharding_strategy=FULL_SHARD \
#     --fsdp_backward_prefetch=BACKWARD_PRE \
#     --fsdp_state_dict_type=SHARDED_STATE_DICT \
#     --mixed_precision=bf16 \
#     $(which lerobot-train) "${TRAIN_ARGS[@]}"

lerobot-train "${TRAIN_ARGS[@]}"
