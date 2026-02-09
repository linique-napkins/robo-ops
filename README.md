# How to run this code!
This is written out for MacOS and Linux. 
If you are on windows... like it should still work, but you will need to install ffmpeg and uv a little differently.

## 1. Install dependencies

You need to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you are on mac,
```bash
brew install ffmpeg
```

## 2. Clone the repo

Install repo into a directory on your laptop. 

There are two ways to do this. I prefer the first since I know it better, but both work:

### Install with SSH

To do this we will use SSH. Install ssh keys with [this guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Then [add it](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Then install it into your laptop.
```bash
mkdir ~/code
cd ~/code
git clone git@github.com:linique-napkins/robo-ops.git
cd robo-ops
```

### Install with GH CLI 

To do this we will use the GitHub CLI. Install it here: https://cli.github.com/

Then install it into your laptop.
```bash
mkdir ~/code
cd ~/code
gh repo clone linique-napkins/robo-ops
```

## 3. Find devices

### Find robot arm ports

Run the motor discovery script and wiggle each arm when prompted:
```bash
uv run utils/find_motors.py
```

This will detect all connected motor controllers, identify each arm by movement, and output stable `/dev/serial/by-id/` paths that persist across reboots. Copy the output into `config.toml`.

### Find cameras

Take a test photo from each configured camera:
```bash
uv run utils/test_cameras.py
```

Photos are saved to `outputs/camera_test/`. Update the `[camera.*]` sections in `config.toml` with the correct device indices or paths. On Linux, USB 2.0 cameras sharing a bus need `fourcc = "MJPG"` to avoid bandwidth issues.

### 3.5. Configure the devices

If on linux you need to add the user to dialout:
```bash
sudo usermod -aG dialout $USER
```

YOU SHOULD NOT NEED TO RUN THIS STEP AS THE MOTORS ARE ALREADY SETUP!
```bash
uv run setup/motor_setup.py
```


## 4. Calibration

```bash
uv run setup/calibrate.py
```
When running this:
- Choose "Bimanual" to calibrate all 4 arms, or "Single arm" for one at a time
- You will need to move every joint until the numbers for min and max stop changing
- It is critical that you do this for all motors


## 4.5. Install HF CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

## 5. Data Collection

### Test teleoperation (without recording)
```bash
uv run data_taking/teleop.py
```

### Record training data
```bash
uv run data_taking/record.py
```

Configure recording settings in `data_taking/config.toml`:
- `repo_id`: HuggingFace dataset repository
- `task`: Task description for the dataset
- `num_episodes`: Number of episodes to record
- `fps`: Recording frame rate

### Replay recorded episodes
```bash
uv run data_taking/replay.py --episode 0
```

## 6. Training

Train an ACT (Action Chunking Transformer) policy on your recorded data:

```bash
uv run training/train.py
```

Configure training in `training/config.toml`:
- `dataset_repo_id`: HuggingFace dataset to train on
- `output_repo_id`: Where to push the trained model
- `steps`: Number of training steps (default: 100,000)
- `batch_size`: Training batch size
- `chunk_size`: ACT action chunk size

Training uses **wandb** for experiment tracking. Disable with `--no-wandb`.

Models are saved to HuggingFace Hub. Disable with `--no-push`.

## 7. Inference

Run a trained policy on the robot:

```bash
uv run inference/run.py
```

Configure in `inference/config.toml`:
- `policy_repo_id`: HuggingFace model to load
- `dataset_repo_id`: Dataset used for training (needed for feature definitions)
- `task`: Task description

Press `q` to stop inference.

## Project Structure

```
.
├── config.toml           # Global hardware config (ports, cameras)
├── lib/
│   └── config.py         # Shared config utilities
├── setup/
│   ├── calibrate.py      # Arm calibration
│   └── motor_setup.py    # Motor configuration
├── utils/
│   ├── find_motors.py    # Motor port discovery
│   ├── health.py         # Motor health check
│   ├── test_cameras.py   # Camera test photos
│   └── test_realsense.py # RealSense camera viewer
├── data_taking/
│   ├── config.toml       # Recording settings
│   ├── record.py         # Data collection with audio cues
│   ├── replay.py         # Replay episodes
│   └── teleop.py         # Test teleoperation
├── training/
│   ├── config.toml       # Training settings
│   └── train.py          # ACT policy training with wandb
└── inference/
    ├── config.toml       # Inference settings
    └── run.py            # Run trained policy on robot
```

## Documentation

- [Adapter Board](https://wiki.seeedstudio.com/xiao_bus_servo_adapter/)
- [HF SO-101](https://huggingface.co/docs/lerobot/so101)
- [Seeed Studio Wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- [Lerobot](https://huggingface.co/docs/lerobot/index)


Motors can run between 5V and 8.4V