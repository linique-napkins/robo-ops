normal uv install
mkdir -p ~/venvs
uv venv ~/venvs/robo-ops

mkdir -p /scratch/ss-engineeringphysics-1/$USER/robo-ops

source ~/venvs/robo-ops/bin/activate
cd /arc/project/ss-engineeringphysics-1/2617-Napkin-Folding/robo-ops
uv sync --active