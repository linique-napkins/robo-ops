# Training on ARC Sockeye

ACT policy training for bimanual SO101 robot on UBC ARC Sockeye HPC cluster.

## TLDR

```bash
# ── Upload (from local machine) ─────────────────────────────
# Sync dataset to Sockeye project storage
rsync -avzP data/datasets/jhimmens/linique \
    jhimmens@dtn.sockeye.arc.ubc.ca:/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/datasets/

# Sync repo to scratch
rsync -avzP --exclude data/ --exclude .venv/ --exclude wandb/ \
    . jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/robo-ops/

# ── Train (on Sockeye login node) ───────────────────────────
cd /scratch/ss-engineeringphysics-1/$USER/robo-ops

sbatch --test-only training/arc_train.sh              # dry run
TRAIN_STEPS=5 sbatch --time=0:15:00 training/arc_train.sh  # quick test
sbatch training/arc_train.sh                          # full training

squeue -u $USER                                       # check job status
tail -f /scratch/ss-engineeringphysics-1/$USER/training_outputs/output-<jobid>.txt
scontrol show job <jobid>

# ── After training (on Sockeye login node) ──────────────────
# Sync wandb offline runs to cloud
wandb sync /scratch/ss-engineeringphysics-1/$USER/robo-ops/wandb/offline-run-*

# Copy checkpoints from scratch to project storage (scratch gets purged!)
cp -r /scratch/ss-engineeringphysics-1/$USER/training_outputs \
    /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/models/

# ── Download (from local machine) ───────────────────────────
# Pull trained model to local machine
rsync -avzP jhimmens@dtn.sockeye.arc.ubc.ca:/scratch/ss-engineeringphysics-1/jhimmens/training_outputs/final \
    data/outputs/jhimmens_linique-act/

# Run inference locally
uv run inference/run.py
```

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

---

## Storage Tiers

| Tier | Path | Quota | Purge | Compute Access |
|------|------|-------|-------|----------------|
| Home | `/home/$USER` | 50 GB | No | Read/Write |
| Project | `/arc/project/ss-engineeringphysics-1/2617-Napkin-Folding` | 5 TB | No | **READ-ONLY** |
| Scratch | `/scratch/ss-engineeringphysics-1/$USER` | 5 TB | **Yes** | Read/Write |

- **Project is read-only on compute nodes.** Dataset lives here, job reads via symlink. All output must go to scratch.
- **Scratch gets purged.** Copy results to project storage after training.

---

## Configuration

Edit `training/config.toml`:

```toml
[training]
dataset_repo_id = "jhimmens/linique"
output_repo_id = "jhimmens/linique-act"
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

---

## Performance Tips

### Mixed Precision (AMP)

V100 Tensor Cores give ~1.5-2x speedup with FP16. **Not yet enabled in `train.py`.**

```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type="cuda"):
    loss, loss_dict = policy.forward(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

### Multi-GPU (Future)

4x V100 DDP cuts training time by ~75%. Requires `train.py` changes (DDP init, distributed sampler).

```bash
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G

torchrun --nproc_per_node=4 training/train.py
```

---

## Troubleshooting

**"CUDA not available"**
- Check `module load cuda` is in the SLURM script
- Verify PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

**OOM (Out of Memory)**
- Reduce `batch_size` in `training/config.toml`
- Enable mixed precision (AMP)
- Check if you landed on a 16 GB node (add `#SBATCH --constraint=gpu_mem_32`)

**wandb fails to sync**
- Compute nodes have no internet — `arc_train.sh` uses `WANDB_MODE=offline`
- Sync from login node: `wandb sync wandb/offline-run-*`

**"Disk quota exceeded"**
- `print_quota` to check usage
- Move uv cache to scratch: `UV_CACHE_DIR=/scratch/.../uv-cache`
- Home is only 50 GB
