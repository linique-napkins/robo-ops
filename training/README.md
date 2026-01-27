# Training

Scripts for training robot manipulation policies.

## Scripts

### Train
Train a policy on collected data:
```bash
uv run training/train.py
```

**Note**: Training is not yet implemented. See [LeRobot documentation](https://huggingface.co/docs/lerobot/index) for training instructions.

## Configuration

Edit `training/config.toml` to configure:
- `repo_id` - Dataset to train on
- `policy` - Policy type (act, diffusion, tdmpc)
- `steps` - Number of training steps
- `batch_size` - Training batch size
- `learning_rate` - Learning rate
- `device` - Device to train on (cuda, mps, cpu)
