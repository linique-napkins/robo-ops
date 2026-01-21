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
brew install ffmpef
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

Find each device.
```bash
uv run lerobot-find-port
```

For each device update the config file at `ports.toml`.

### 3.5. Configure the devices

YOU SHOULD NOT NEED TO RUN THIS STEP AS THE MOTOR ARE ALREADY SETUP!
```bash
uv run utils/motor-setiup.py
```


## 4. Calibration

```bash
uv run utils/calibrate.py
```
When running this. 
You will need to move every joint until the numbers for min and max stop changing. 
It is critical that you do this for all motors.
Repeat as necessary for all 4 arms

## 5. Teleoperation

Set the arms you want to control in `teleop.py` then run:

```bash
uv run utils/leader-teleop.py
```

## Documentation:

- [Adapter Board](https://wiki.seeedstudio.com/xiao_bus_servo_adapter/)
- [HF SO-101](https://huggingface.co/docs/lerobot/so101)
- [Seeed Studio Wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)