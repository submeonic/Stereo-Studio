#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e '.[gui,test]'
if ! command -v ffmpeg >/dev/null; then
  echo 'For MP4 export, install FFmpeg: sudo apt install ffmpeg'
fi
printf '\nReady. Start the app with ./run.sh\n'
