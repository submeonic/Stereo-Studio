# Stereo Studio

Create looping 3D wigglegrams from Fujifilm W3 or Nintendo 3DS MPO photographs, or a pair of left/right images. Stereo Studio is a local Python desktop app with a command-line interface.

- **Smooth 3D motion** synthesizes intermediate viewpoints; **Classic two-frame** alternates the original eyes.
- **Anaglyph 3D** exports a red/cyan JPEG for glasses.
- Click an anchor to hold a subject steady, adjust motion, edit colors and crop, and add independent temporal effects.
- Save your own Photo Edit and Effects presets as portable JSON files. No artistic presets are bundled.
- CPU stereo works without a discrete GPU. Optional NVIDIA Fast-FoundationStereo provides another geometry estimator.
- Photo processing runs locally. GPU model downloads happen only during explicit setup.

## Start here

1. Follow [installation for Linux or Windows](docs/SETUP.md), choosing CPU or optional GPU setup.
2. Follow [your first wigglegram](docs/WORKFLOW.md).
3. Use the [settings and effects glossary](docs/GLOSSARY.md) when choosing controls. Hover over main-setting labels or inputs for help.

After setup:

```bash
# Linux / WSL
./run.sh
```

```powershell
# Native Windows CPU
.\run.ps1
```

Default desktop settings: **7 views at 20 FPS**, 1280 px geometry, **960 px preview**, **1280 px export**, **preview frame AA on**, and **2× export supersampling**. Preview and export quality are independent. Source resolution and crop limit recoverable detail.

## Input and output

Use the MPO containing both eyes, rather than its accompanying single-eye JPEG. Select **Nintendo 3DS · nominal** for 3DS photos. This uses image-based alignment; it is not a measured camera calibration. W3/3DS nominal profiles do not produce metric depth. The app does not invent a camera's lens parameters or baseline.

Export MP4 for full-color animation, GIF for a looping image, or JPEG for anaglyph mode. MP4 needs FFmpeg with the `libx264` encoder on PATH. Originals stay untouched. Generated output defaults to `outputs/`, which Git ignores.

## CLI examples

```bash
# Linux / WSL; on Windows replace .venv/bin/python with .\.venv\Scripts\python.exe
.venv/bin/python -m wigglegram inspect photo.mpo
.venv/bin/python -m wigglegram render photo.mpo --backend sgbm -o outputs/wiggle.mp4
.venv/bin/python -m wigglegram render photo.mpo --backend neural --width 960 --output-width 1280 --supersample 2 -o outputs/gpu.mp4
.venv/bin/python -m wigglegram render photo.mpo --profile 3ds --mode classic -o outputs/classic.gif
.venv/bin/python -m wigglegram render photo.mpo --mode anaglyph --aspect 1 -o outputs/anaglyph.jpg
.venv/bin/python -m wigglegram render --help
```

CLI export defaults to 1280 px with 2× supersampling. `--width` controls cached analysis, not final output size. The CLI's optional `--antialias 2x` is an alternative to export supersampling: use `--supersample 1` with it to avoid two supersampling passes. Desktop preview AA is separate and does not add another pass during export.

Optional [stereo asset bundles and measured camera calibration](docs/ASSET_FORMAT.md) support reuse of aligned eyes, disparity, and settings. [Third-party notes](docs/THIRD_PARTY.md) describe the optional model and its separate terms.

## Limits and support

Hair, foliage, reflections, moving subjects, occlusions, and covered lenses can produce errors. Smaller sweeps often help. Local edge repair does not blur the whole frame, but cannot recover every missing surface. The Geometry tab provides depth/disparity, source projections, and repair diagnostics.

Linux CPU and NVIDIA rendering have been exercised locally. Windows CPU setup and a Windows/Linux CPU CI workflow are included; Windows execution and WSL GPU execution require validation on those systems. The Nintendo 3DS nominal path is tested with synthetic MPO inputs; real 3DS capture validation remains outstanding.

For a bug report, include operating system, Python version, backend, mode, geometry/preview/export sizes, and the error text. Share a sample only if you have permission to distribute it.

## Development checks

```bash
.venv/bin/python -m pip install -e '.[gui,test]'
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Tests generate synthetic images; no personal sample collection is required. Local photos, saved presets, model weights, caches, AI instructions, and private development notes are excluded from the publication branch.
