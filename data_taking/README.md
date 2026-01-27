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

## Recording Controls

During recording:
- **`e`** - End current episode early (saves it)
- **`r`** - Re-record current episode (discards and restarts)
- **`q`** - Stop recording session
