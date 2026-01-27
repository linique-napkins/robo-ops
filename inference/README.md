# Inference

Scripts for running trained policies on the robot.

## Scripts

### Run
Run inference with a trained policy:
```bash
uv run inference/run.py
```

**Note**: Inference is not yet implemented. See [LeRobot documentation](https://huggingface.co/docs/lerobot/index) for inference instructions.

## Configuration

Edit `inference/config.toml` to configure:
- `policy_path` - Path to trained policy checkpoint
- `device` - Device to run on (cuda, mps, cpu)
- `display` - Whether to show visualization
