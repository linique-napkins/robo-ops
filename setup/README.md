# Setup

Scripts for initial robot hardware setup and calibration.

## Scripts

### Motor Setup
Configure motor parameters (run once when setting up new hardware):
```bash
uv run setup/motor_setup.py
```

### Calibration
Calibrate robot arms for teleoperation:
```bash
# Interactive mode - choose single arm or bimanual
uv run setup/calibrate.py

# Calibrate all 4 arms at once
uv run setup/calibrate.py --bimanual

# Force recalibration (overwrite existing)
uv run setup/calibrate.py --bimanual --force
```

## When to Run

1. **Motor Setup**: Only needed once per arm when first setting up hardware
2. **Calibration**: Run whenever:
   - Setting up a new arm
   - Robot behavior seems off
   - After hardware changes
