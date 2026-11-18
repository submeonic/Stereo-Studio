#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=476f4249561f7c79ca707326954f9255643412a6
if [[ ! -d third_party/Fast-FoundationStereo ]]; then
  git clone https://github.com/NVlabs/Fast-FoundationStereo.git third_party/Fast-FoundationStereo
  git -C third_party/Fast-FoundationStereo checkout "$revision"
fi
if [[ $(git -C third_party/Fast-FoundationStereo rev-parse HEAD) != "$revision" ]]; then
  echo "Expected upstream revision $revision. Preserve local edits before changing the checkout." >&2
  exit 1
fi
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -r requirements-gpu.lock --extra-index-url https://download.pytorch.org/whl/cu124
.venv-gpu/bin/python scripts/download_model.py
printf '\nGPU backend installed. Run ./run.sh and select GPU stereo.\n'
