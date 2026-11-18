"""Explicit setup-time download from the authors' linked Google Drive checkpoints."""
"""Explicit setup-time download from the authors' linked Google Drive checkpoints."""

import hashlib
from pathlib import Path

import gdown

root = Path(__file__).resolve().parents[1]
out = root / "models/fast-foundation-stereo/23-36-37"
out.mkdir(parents=True, exist_ok=True)
files = [
    ("cfg.yaml", "1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj"),
    ("model_best_bp2_serialize.pth", "1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E"),
]
for name, identifier in files:
    target = out / name
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".download")
        gdown.download(id=identifier, output=str(temporary))
        if name.endswith(".pth"):
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != "af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990":
                raise RuntimeError("Downloaded checkpoint checksum differs from the tested model.")
        temporary.replace(target)
model = out / "model_best_bp2_serialize.pth"
if (
    hashlib.sha256(model.read_bytes()).hexdigest()
    != "af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990"
):
    raise RuntimeError("Installed checkpoint checksum differs from the tested model.")
print("Verified official model:", model)
