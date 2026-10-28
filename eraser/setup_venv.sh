#!/bin/bash
# Without Docker: a Python 3.10 virtual environment with the reviewed versions.
#   bash setup_venv.sh /opt/tl-venv
set -euo pipefail
VENV=${1:-./.venv}
cd "$(dirname "$0")"
${PY:-python3.10} -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install -r requirements.txt
"$VENV/bin/python" - <<'PY'
import sys, torch, diffusers, transformers, bitsandbytes
sys.path.insert(0, "tl"); sys.path.insert(0, "tl/third_party/lama")
from saicinpainting.training.trainers import load_checkpoint     # LaMa's whole import chain
from transformers import Sam2Model, Sam2Processor
print("imports ok; torch", torch.__version__, "cuda available", torch.cuda.is_available())
PY
echo "done: $VENV"
