# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Overview

This is a robotics training data collection project using LeRobot with SO101 robot arms.
It controls leader/follower arm pairs for teleoperation and data recording.

## Tech Stack

- **Python**: 3.13+
- **Package Manager**: uv (not pip)
- **Linting**: ruff
- **Type Checking**: ty
- **CLI Framework**: typer
- **Robot SDK**: lerobot

## Build & Run Commands

### Install Dependencies

```bash
uv sync
```

### Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
uv run ty check
```

### Run Scripts

Scripts are run directly with uv, not installed as commands:

```bash
uv run utils/motor-setup.py
uv run utils/calibrate.py
uv run utils/teleop.py
```

### Find Robot Ports

```bash
uv run lerobot-find-port
```

## Code Style Guidelines

### Imports

- **One import per line** (enforced by isort with `force-single-line = true`)
- **Order**: stdlib, third-party, local (separated by blank lines)
- **No unused imports** (enforced by ruff F401)

```python
# Good
import tomllib
from pathlib import Path

import typer
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig

# Bad - multiple imports on one line
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
```

### Formatting

- **Line length**: 100 characters max
- **Quotes**: Double quotes for strings
- **Indentation**: 4 spaces (no tabs)
- **Trailing commas**: Preserved

### Type Hints

- Add return type hints to all functions
- Use `dict`, `list`, `set` directly (not `typing.Dict`, etc. - Python 3.13)

```python
def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)

def get_arm_choice() -> str:
    ...
```

### Pathlib

- **Always use `pathlib.Path`** instead of `os.path` (enforced by PTH rules)
- Use `Path.open()` instead of built-in `open()`

```python
# Good
from pathlib import Path
CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"
with CONFIG_PATH.open("rb") as f:
    ...

# Bad
import os
config_path = os.path.join(os.path.dirname(__file__), "..", "ports.toml")
with open(config_path, "rb") as f:
    ...
```

### Naming Conventions

- **Variables/functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Classes**: PascalCase

```python
CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"
ARM = "right"

def get_arm_choice() -> str:
    ...
```

### Error Handling

- For CLI scripts using typer, use `typer.Exit(code)` for early exits
- Validate configuration early and provide clear error messages

```python
if not port:
    typer.echo(f"\nError: No port configured for {arm} {role}.")
    typer.echo("Please update ports.toml with the correct port.")
    raise typer.Exit(1)
```

### Configuration

- Robot configuration lives in `ports.toml` at project root
- Structure: `[leader.right]`, `[leader.left]`, `[follower.right]`, `[follower.left]`
- Each has `port` and `id` fields

```toml
[leader.right]
port = "/dev/tty.usbmodem5AB90657441"
id = "leader_right"
```

### CLI Scripts

- Use typer for interactive CLI tools
- Keep prompts simple and non-technical for end users
- Always confirm destructive/hardware operations before proceeding

```python
app = typer.Typer()

@app.command()
def main():
    """Brief description of what this does."""
    ...

if __name__ == "__main__":
    app()
```

## Ruff Rules Enabled

The following rule sets are enforced (see `pyproject.toml`):

- `E`, `W`: pycodestyle errors/warnings
- `F`: pyflakes
- `I`: isort
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `UP`: pyupgrade
- `ARG`: unused arguments
- `SIM`: simplify
- `TCH`: type-checking imports
- `PTH`: use pathlib
- `ERA`: no commented-out code
- `PL`: pylint
- `PERF`: performance
- `RUF`: ruff-specific

## Project Structure

```
.
├── ports.toml          # Robot port/ID configuration
├── pyproject.toml      # Project config, dependencies, tool settings
├── main.py             # Main entry point (currently empty)
└── utils/
    ├── calibrate.py    # Interactive arm calibration CLI
    ├── motor-setup.py  # Interactive motor setup CLI
    └── teleop.py       # Teleoperation script with live position display
```

## Important Notes

- Always run `uv run ruff check .` and `uv run ty check` before committing
- Do not use pip; this project uses uv exclusively
- Hardware scripts require physical robot arms to be connected
- Port paths in `ports.toml` are macOS-specific (`/dev/tty.usbmodem*`)
