# Training on ARC Sockeye

ACT policy training for bimanual SO101 robot, targeting UBC ARC Sockeye HPC cluster.

# ARC

```bash 
# Dry run — validates SBATCH config, no GPU time used                 
sbatch --test-only training/arc_train.sh
                                                                                                                                                                                                                    
# Real test — 5 training steps, 15 min wall time, minimal allocation burn  
TRAIN_STEPS=5 sbatch --time=0:15:00 training/arc_train.sh

# full training
sbatch training/arc_train.sh
``` 

## Quick Reference

```bash
# Local (macOS/Linux with GPU)
uv run training/train.py

# On Sockeye
sbatch training/sockeye_train.sh
squeue -u $USER                    # check job status
scancel <jobid>                    # cancel job
```

## Plan Overview

1. Transfer dataset + code to Sockeye
2. Set up Python environment with uv on the cluster
3. Submit SLURM GPU job
4. Monitor with wandb
5. Pull trained model back to local machine

---

## Sockeye GPU Hardware

| Spec | Details |
|------|---------|
| GPU model | NVIDIA Tesla V100-SXM2 |
| GPU memory | **32 GB** (44 nodes) or 16 GB (6 nodes) |
| GPUs per node | 4 |
| CPU cores per node | 24 (= 6 per GPU) |
| RAM per node | 192 GB |
| Interconnect | InfiniBand EDR 100 Gbps |
| Local scratch | 1.9 TB NVMe per node |
| Total GPUs | 200 across 50 nodes |

The V100 supports CUDA Compute Capability 7.0, all modern CUDA toolkits, and has Tensor Cores for FP16 acceleration.

---

## Storage Tiers (Critical)

| Tier | Path | Quota | Purge | Backup | Compute Access |
|------|------|-------|-------|--------|----------------|
| Home | `/home/$USER` | 50 GB | No | Yes | Read/Write |
| Project | `/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding` | 5 TB | No | Limited | **READ-ONLY** |
| Scratch | `/scratch/st-<pi>-1/$USER` | 5 TB | **Yes** | No | Read/Write |

### The Big Gotcha: Project is Read-Only on Compute Nodes

Your training job **cannot write** to `/arc/project/`. All output (checkpoints, wandb cache, logs) must go to `/scratch/`. This means:

- Dataset: store in project (persistent), job reads from there
- Checkpoints: write to scratch during training
- After job: copy important results from scratch back to project

```bash
# After training completes, save results to project
cp -r /scratch/st-<pi>-1/$USER/outputs/ /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/models/
```

### Scratch Gets Purged

Files on scratch can be deleted without warning when disk space is low. Move results to project storage promptly after jobs finish.

---

## Environment Setup

### Installing uv on Sockeye

Standard `curl | sh` install tries to modify shell profile. Use a controlled install instead:

```bash
# On login node
wget https://astral.sh/uv/install.sh
mkdir -p ${HOME}/uv
UV_INSTALL_DIR="${HOME}/uv" INSTALLER_NO_MODIFY_PATH=1 ./install.sh
echo 'export PATH="${HOME}/uv:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Python + PyTorch Version Matrix

The V100 works with both CUDA 11.x and 12.x. Match PyTorch to whatever CUDA module Sockeye has:

| PyTorch | CUDA | Status |
|---------|------|--------|
| 2.5.1 | 11.8 | Safe, well-tested |
| 2.6.0 | 12.4 / 12.6 | Good if CUDA 12 available |
| 2.7+ | 12.6+ | Latest, check compatibility |

Check what is available:
```bash
module avail cuda
module avail python
```

### Why uv Needs Special Handling on HPC

**Problem 1: Python version.** This project requires Python 3.13+, but Sockeye's `module load python` may only have 3.8-3.11. Options:

```bash
# Option A: Let uv manage Python (preferred)
uv python install 3.13
uv sync

# Option B: Use a module if available
module load python/3.13
uv venv --python $(which python)
```

**Problem 2: PyTorch CUDA wheels.** uv needs to pull the CUDA-specific PyTorch build, not the CPU-only default. The project's `pyproject.toml` currently doesn't specify a PyTorch CUDA index because LeRobot handles it. On Sockeye, you may need to force it:

```bash
# After uv sync, reinstall torch with CUDA support
module load cuda/11.8  # or whatever version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --reinstall
```

Or configure in `pyproject.toml` for Sockeye builds:
```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu118", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu118", marker = "sys_platform == 'linux'" }]
```

**Problem 3: Inode limits.** Python venvs and uv cache create thousands of small files, which can exhaust scratch's 10M inode limit. Redirect cache:

```bash
export UV_CACHE_DIR=/scratch/st-<pi>-1/$USER/.uv-cache
export UV_PROJECT_ENVIRONMENT=/scratch/st-<pi>-1/$USER/.venv
```

**Problem 4: No internet on compute nodes.** Compute nodes may not have network access. Install all packages on the login node first, then jobs use the cached venv.

### Full Environment Setup (Run Once on Login Node)

```bash
ssh jhimmens@sockeye.arc.ubc.ca

# Set up paths
export ALLOC=st-<pi>-1
export PROJECT=/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding
export SCRATCH=/scratch/$ALLOC/$USER
mkdir -p $SCRATCH

# Clone repo to scratch (writable)
cd $SCRATCH
git clone --recurse-submodules <repo-url> robo-ops
cd robo-ops

# Set up uv environment
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv
module load cuda  # load CUDA before installing torch

uv sync
# Verify CUDA torch
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## Data Transfer

### Upload Dataset to Sockeye

Use the data transfer nodes (DTN), not login nodes, for large transfers:

```bash
# From local machine
scp -r data/datasets/jhimmens/linique \
    jhimmens@dtn.sockeye.arc.ubc.ca:/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/datasets/

# Or use rsync (better for large/resumable transfers)
rsync -avzP data/datasets/jhimmens/linique \
    jhimmens@dtn.sockeye.arc.ubc.ca:/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/datasets/
```

### Download Trained Model

```bash
# From local machine
scp -r jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/st-<pi>-1/jhimmens/robo-ops/data/outputs/jhimmens_linique-act/final \
    data/outputs/jhimmens_linique-act/
```

---

## SLURM Job Script

Create `training/sockeye_train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=linique-act
#SBATCH --account=st-<pi>-gpu          # GPU allocation (ask PI for exact code)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6              # 6 cores per GPU (24 total / 4 GPUs)
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1             # Start with 1 GPU
#SBATCH --time=24:00:00               # 24h wall time
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=jhimmens@student.ubc.ca
#SBATCH --mail-type=ALL

# ── Environment ──────────────────────────────────────────────
module load gcc cuda

export ALLOC=st-<pi>-1
export SCRATCH=/scratch/$ALLOC/$USER
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv
export PATH="${HOME}/uv:$PATH"

# wandb offline mode if compute nodes lack internet
# export WANDB_MODE=offline

cd $SCRATCH/robo-ops

# ── Symlink dataset from project (read-only) to local path ──
# train.py expects data at data/datasets/<repo_id>
mkdir -p data/datasets/jhimmens
ln -sfn /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/datasets/linique \
    data/datasets/jhimmens/linique

# ── Train ────────────────────────────────────────────────────
uv run training/train.py --no-push --wandb
```

Submit:
```bash
sbatch training/sockeye_train.sh
```

Monitor:
```bash
squeue -u $USER          # job status
tail -f linique-act-*.out  # live output
sacct -j <jobid>          # after completion
```

---

## Configuration

Edit `training/config.toml`:

```toml
[training]
dataset_repo_id = "jhimmens/linique"
output_repo_id = "jhimmens/linique-act"
policy = "act"

steps = 100000
batch_size = 8        # V100 32GB can handle 16-32 depending on image size
learning_rate = 1e-5
device = "cuda"

chunk_size = 100
dim_model = 512
n_heads = 8
n_encoder_layers = 4

save_freq = 10000
log_freq = 100

wandb_project = "linique-robot"
```

### Batch Size Guidance for V100-32GB

With ACT (dim_model=512, chunk_size=100, 3 cameras at 640x480):
- `batch_size = 8`: ~12 GB VRAM, safe starting point
- `batch_size = 16`: ~20 GB VRAM, good throughput
- `batch_size = 32`: ~28 GB VRAM, near the limit

If you hit OOM, reduce batch size or enable gradient accumulation (not yet in train.py but easy to add).

---

## Performance Tips

### Mixed Precision (AMP)

The V100 has Tensor Cores optimized for FP16. Using automatic mixed precision gives ~1.5-2x speedup and halves memory usage. **This is not yet enabled in `train.py` but should be.** The change is small:

```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type="cuda"):
    loss, loss_dict = policy.forward(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

### DataLoader Workers

Each GPU node has 24 cores across 4 GPUs = 6 cores per GPU. Current `num_workers=4` is good. Don't exceed 5.

### Multi-GPU (Future)

For multi-GPU on a single node (up to 4x V100), wrap the model with `DistributedDataParallel`:

```bash
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G

torchrun --nproc_per_node=4 training/train.py
```

This requires changes to `train.py` (DDP init, distributed sampler). Benchmark results on Sockeye show 4x DDP cuts training time by ~75% vs single GPU.

| Setup | GPUs | Time (ResNet50, 5 epochs) |
|-------|------|---------------------------|
| Single | 1 | 363s |
| DataParallel | 4 | 103s |
| DDP | 4 | 94s |
| DDP multi-node | 8 | 56s |

### Compile the Model (PyTorch 2.0+)

```python
policy = torch.compile(policy)  # can give 10-30% speedup on V100
```

---

## wandb Integration

Training already logs to wandb. On Sockeye:

```bash
# On login node (has internet), authenticate once
wandb login

# If compute nodes lack internet, use offline mode
export WANDB_MODE=offline
# After job, sync from login node:
wandb sync $SCRATCH/robo-ops/wandb/offline-run-*
```

Resume a crashed/timed-out run:
```bash
uv run training/train.py --resume <wandb-run-id>
```

---

## HuggingFace Hub

Push final model after training:
```bash
# From login node (has internet)
uv run training/train.py --push
# Or push a checkpoint manually:
# huggingface-cli upload jhimmens/linique-act data/outputs/jhimmens_linique-act/final
```

---

## Workflow Checklist

- [ ] Transfer dataset to Sockeye project storage via DTN
- [ ] SSH to login node, clone repo to scratch
- [ ] Install uv, set up venv, verify `torch.cuda.is_available()`
- [ ] Confirm CUDA version matches PyTorch build (`python -c "import torch; print(torch.version.cuda)"`)
- [ ] Update `training/config.toml` (batch_size, steps, etc.)
- [ ] Fill in SLURM `--account` with actual allocation code
- [ ] `wandb login` on login node
- [ ] `sbatch training/sockeye_train.sh`
- [ ] Monitor with `squeue`, `tail -f`, wandb dashboard
- [ ] After completion: copy checkpoints from scratch to project storage
- [ ] Pull trained model to local machine
- [ ] Test inference locally: `uv run inference/run.py`

---

## Troubleshooting

**"CUDA not available" in job output**
- Check `module load cuda` is in your SLURM script
- Verify PyTorch was installed with CUDA support, not CPU-only
- Run `nvidia-smi` in an interactive job to confirm GPUs are visible

**OOM (Out of Memory)**
- Reduce `batch_size` in `training/config.toml`
- Enable mixed precision (AMP) -- halves memory
- Check if you landed on a 16 GB V100 node (6 of 50 nodes)

**Job stuck in queue**
- GPU jobs compete for 50 nodes. Request less time or fewer GPUs
- Check allocation balance: `print_quota`
- Try off-peak hours

**wandb fails to sync**
- Compute nodes may lack internet. Use `WANDB_MODE=offline` and sync later
- `wandb sync wandb/offline-run-*` from login node

**uv can't find packages**
- Compute nodes may lack internet. Install everything on login node first
- Ensure `UV_PROJECT_ENVIRONMENT` points to scratch venv

**"Disk quota exceeded"**
- Check with `print_quota`
- Move uv cache to scratch: `UV_CACHE_DIR=/scratch/.../uv-cache`
- Clean cache: `uv cache clean`
- Remember home is only 50 GB

**Python version mismatch**
- Project requires 3.13+, Sockeye may not have it as a module
- Use `uv python install 3.13` to let uv manage Python
