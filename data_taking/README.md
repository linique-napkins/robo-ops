# Data Taking

Scripts for collecting teleoperation data for training.

## Scripts

### Record
Record episodes of teleoperation data:
```bash
uv run data_taking/record.py
```

Options:
- `--no-push` - Don't upload to Hugging Face Hub after recording
- `--no-display` - Disable Rerun visualization
- `--no-resume` - Create new dataset instead of resuming

### Replay
Replay a recorded episode on the robot:
```bash
uv run data_taking/replay.py --episode 0
uv run data_taking/replay.py --episode 0 --repo-id jhimmens/linique
```

### Teleop
Test teleoperation without recording:
```bash
uv run data_taking/teleop.py
```

## Configuration

Edit `data_taking/config.toml` to change recording settings:
- `repo_id` - Hugging Face dataset repository
- `task` - Task description
- `num_episodes` - Number of episodes to record
- `episode_time` - Duration of each episode (seconds)
- `reset_time` - Time for environment reset between episodes

## Pushing to Hugging Face Hub

Upload the local dataset to HuggingFace:
```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
root = Path('data/datasets/jhimmens/linique-v2')
ds = LeRobotDataset('jhimmens/linique-v2', root=root)
ds.push_to_hub()
"
```

Tag a specific version after pushing:
```bash
uv run huggingface-cli tag jhimmens/linique-v2 v1.0 --repo-type dataset
```

## Recording Controls

During recording:
- **`e`** - End current episode early (saves it)
- **`r`** - Re-record current episode (discards and restarts)
- **`q`** - Stop recording session
