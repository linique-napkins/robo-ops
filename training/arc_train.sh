#!/bin/bash

#SBATCH --job-name=ENPH-2617-Linique-TRAIN
#SBATCH --account=engineeringphysics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=gpu_mem_32
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --output=output-%j.txt
#SBATCH --error=error-%j.txt
#SBATCH --mail-user=jhimmens@student.ubc.ca
#SBATCH --mail-type=ALL

# ── Paths ───────────────────────────────────────────────────
ALLOC=st-engineeringphysics-1
SCRATCH=/scratch/$ALLOC/$USER
PROJECT=/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding
OUTPUT_DIR=$SCRATCH/training_outputs

# ── Environment ─────────────────────────────────────────────
module load gcc
module load cuda

source $HOME/.venv/bin/activate

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Sync dataset from project storage (read-only) ──────────
# train.py expects data at data/datasets/<repo_id>
mkdir -p data/datasets/jhimmens
ln -sfn $PROJECT/datasets/linique data/datasets/jhimmens/linique

# ── Output dir on scratch (writable) ───────────────────────
mkdir -p $OUTPUT_DIR

# ── Train ───────────────────────────────────────────────────
# Pass TRAIN_STEPS env var to override steps (e.g. TRAIN_STEPS=5 for testing)
STEPS_FLAG=""
if [ -n "$TRAIN_STEPS" ]; then
    STEPS_FLAG="--steps $TRAIN_STEPS"
fi

python training/train.py --yes --no-push --output-dir $OUTPUT_DIR $STEPS_FLAG
