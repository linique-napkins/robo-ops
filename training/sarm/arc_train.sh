#!/bin/bash

#SBATCH --job-name=ENPH-2617-Linique-SARM
#SBATCH --account=ss-engineeringphysics-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=gpu_mem_32
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/ss-engineeringphysics-1/%u/training_outputs/sarm-output-%j.txt
#SBATCH --error=/scratch/ss-engineeringphysics-1/%u/training_outputs/sarm-error-%j.txt
#SBATCH --mail-user=jhimmens@student.ubc.ca
#SBATCH --mail-type=ALL

# ── Paths ───────────────────────────────────────────────────
ALLOC=ss-engineeringphysics-1
SCRATCH=/scratch/$ALLOC/$USER
PROJECT=/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding
OUTPUT_DIR=$SCRATCH/training_outputs/sarm

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
export HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export WANDB_DIR=$SCRATCH/robo-ops

# ── Pre-download CLIP weights (cached in TORCH_HOME) ───────
# SARM uses CLIP for image/text encoding. Pre-download on login node if needed:
#   python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
# Weights are cached at $HF_HOME/hub/models--openai--clip-vit-base-patch32/

# ── Sync dataset from project storage (read-only) ──────────
# Read dataset_repo_id from SARM config
DATASET_REPO_ID=$(grep '^dataset_repo_id' training/sarm/config.toml | sed 's/.*= *"\(.*\)"/\1/')
DATASET_NAME=${DATASET_REPO_ID##*/}   # e.g. "linique-v2"
DATASET_OWNER=${DATASET_REPO_ID%%/*}  # e.g. "jhimmens"

mkdir -p "data/datasets/$DATASET_OWNER"
ln -sfn "$PROJECT/datasets/$DATASET_NAME" "data/datasets/$DATASET_REPO_ID"

# ── Output dir on scratch (writable) ───────────────────────
mkdir -p $OUTPUT_DIR

# ── GPU info ──────────────────────────────────────────────────
nvidia-smi

# ── Train SARM ─────────────────────────────────────────────
# Pass TRAIN_STEPS env var to override steps (e.g. TRAIN_STEPS=5 for testing)
STEPS_FLAG=""
if [ -n "$TRAIN_STEPS" ]; then
    STEPS_FLAG="--steps $TRAIN_STEPS"
fi

python training/sarm/train.py --yes --no-push --output-dir $OUTPUT_DIR $STEPS_FLAG
