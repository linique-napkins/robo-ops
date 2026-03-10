# /// script
# dependencies = [
#   "transformers",
#   "torch",
#   "torchvision",
# ]
# ///
"""Pre-download CLIP weights for offline use on Sockeye compute nodes."""

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

from transformers import CLIPModel  # noqa: E402
from transformers import CLIPProcessor  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"

print(f"Downloading {MODEL_ID} to {os.environ['HF_HOME']}...")
CLIPModel.from_pretrained(MODEL_ID)
CLIPProcessor.from_pretrained(MODEL_ID)
print("Done. Weights cached for offline use.")
