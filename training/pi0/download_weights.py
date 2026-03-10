# /// script
# dependencies = [
#   "huggingface-hub",
# ]
# ///
"""Pre-download Pi0 base weights and PaliGemma tokenizer for offline Sockeye use."""

import os
from pathlib import Path

# Preserve HF token before overriding HF_HOME (login saves to default location)
_default_hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
_token_path = _default_hf_home / "token"
if _token_path.exists() and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = _token_path.read_text().strip()

# Match HF_HOME from arc_train.sh so compute nodes find the cached weights
ALLOC = "ss-engineeringphysics-1"
SCRATCH = Path(f"/scratch/{ALLOC}/{os.environ['USER']}")
os.environ["HF_HOME"] = str(SCRATCH / ".cache/huggingface")

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = [
    "lerobot/pi0_base",
    "google/paligemma-3b-pt-224",
]

for model_id in MODELS:
    print(f"Downloading {model_id} to {os.environ['HF_HOME']}...")
    snapshot_download(model_id)

print("Done. Weights cached for offline use.")
