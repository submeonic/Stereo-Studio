# Installation

## Choose a route

| System | CPU SGBM | NVIDIA Fast-FoundationStereo |
| --- | --- | --- |
| Linux x86-64 | Native desktop; tested locally | Separate Linux GPU environment; tested locally |
| Windows 10/11, 64-bit | Native Python/Qt; setup provided, pending Windows validation | Use WSL2 with Ubuntu and WSLg; pending WSL hardware validation |
| Other hardware | CPU may work where Python/Qt/OpenCV wheels exist | This integration does not target AMD, Intel, or Apple GPU inference |

Use **64-bit Python 3.12** as the reference version. Python, Git, and FFmpeg are separate prerequisites. CPU setup does not download the GPU model. Internet is needed for installation; processing installed inputs does not need it.

## Linux CPU

The following commands target Ubuntu/Pop!_OS. Other distributions need equivalent Python venv, Qt graphics, and FFmpeg packages.

```bash
sudo apt update
sudo apt install git python3 python3-venv ffmpeg libegl1 libopengl0
python3 --version
# Use Python 3.12 for the supplied reference dependency lock.
git clone https://github.com/submeonic/Stereo-Studio.git
cd Stereo-Studio
./scripts/setup.sh
./run.sh
```

If `python3` is not 3.12, create the environment explicitly with a Python 3.12 installation:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e '.[gui]'
./run.sh
```

Choose **CPU · SGBM**. Classic and Anaglyph usually use lightweight sparse alignment unless stereo assets are requested. Smooth mode estimates dense geometry.

## Windows CPU (PowerShell)

Install Git for Windows and [64-bit Python 3.12 with the Python launcher](https://www.python.org/downloads/windows/). Install an FFmpeg build with `libx264` and add its `bin` directory to PATH if you want MP4. The [FFmpeg download page](https://ffmpeg.org/download.html) links Windows build providers. Reopen PowerShell after changing PATH.

```powershell
py -3.12 --version
git clone https://github.com/submeonic/Stereo-Studio.git
cd Stereo-Studio
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

The execution-policy argument applies only to that PowerShell process; no permanent policy change is required. If you prefer not to use scripts:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
.\.venv\Scripts\python.exe -m wigglegram gui
```

Choose **CPU · SGBM**. Native Windows uses the package dependency ranges; `requirements.lock` records the reference Linux environment rather than a Windows-tested lock. No virtual-environment activation is necessary in either route.

## Optional Linux NVIDIA GPU backend

First complete CPU/desktop setup. You need a working NVIDIA driver, a CUDA-capable GPU supported by the pinned PyTorch 2.6.0/CUDA 12.4 runtime, sufficient VRAM, Git, and disk space for a second environment and model. Start with 960 px geometry. The local reference GPU has 16 GB VRAM; this is a tested configuration, not a guaranteed minimum. Very new GPUs may need a newer PyTorch build that this adapter has not validated.

```bash
nvidia-smi
./scripts/setup-gpu.sh
.venv-gpu/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
./run.sh
```

The setup script checks out a pinned upstream revision, installs the separate GPU dependencies, and downloads/verifies the official checkpoint. Review [upstream source and model terms](https://github.com/NVlabs/Fast-FoundationStereo) and [third-party notes](THIRD_PARTY.md). No model weights or upstream code are bundled in this repository. The application continues to run from `.venv`; it launches inference through `.venv-gpu`.

Select **GPU · Fast-FoundationStereo**. If inference fails, the error is shown; it never silently substitutes CPU output. Try lower geometry width or switch to CPU. The installed PyTorch wheel supplies its CUDA runtime; this adapter uses the PyTorch volume-building path and does not require a separately compiled custom CUDA extension.

## Windows NVIDIA GPU via WSL2

Use WSL2 Ubuntu and its Linux Python environments, rather than installing the Linux GPU lock into native Windows. NVIDIA documents [CUDA support through the Windows driver in WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Microsoft documents [Linux GUI applications through WSLg](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

1. Install a compatible NVIDIA **Windows** driver and WSL2/Ubuntu. Follow those official guides for OS requirements and initial setup. Do not install a Linux display driver over the WSL-provided driver.
2. In Ubuntu, verify `nvidia-smi`, then follow both Linux sections above. Clone into the Linux home directory for better filesystem performance; do not reuse a Windows-created `.venv`.
3. Run `./run.sh` inside Ubuntu; WSLg displays the Qt window on the Windows desktop. Windows photos are accessible under paths such as `/mnt/c/Users/<you>/Pictures`.

This is the documented compatibility route, not a claim of a completed WSL hardware test. If CUDA or GUI integration is unavailable, use the native Windows CPU route.

## Check FFmpeg and troubleshoot

```bash
ffmpeg -version
ffmpeg -hide_banner -encoders
```

Look for `libx264`. GIF and JPEG export do not require FFmpeg. If MP4 fails, check PATH, encoder availability, output permissions, and disk space.

- **Linux Qt platform error:** install the distribution's Qt runtime libraries. On a Wayland session try `QT_QPA_PLATFORM=wayland ./run.sh`; on X11 try `QT_QPA_PLATFORM=xcb ./run.sh`. Do not use `offscreen` for interactive work.
- **GPU option unavailable:** run GPU setup from the repository root. Check `.venv-gpu`, `third_party/Fast-FoundationStereo`, and `models/`. Custom paths can be supplied with `WIGGLE_GPU_PYTHON`, `WIGGLE_FFS_REPO`, and `WIGGLE_FFS_MODEL`.
- **Out of memory or slow preview:** reduce geometry width, preview width, or views; disable preview AA temporarily. Export supersampling is independent.
- **No stereo pair in an input:** open the MPO, or choose the two eye images separately. A single JPEG cannot supply stereo geometry.
- **Update:** close the application, pull the reviewed release, then rerun the matching setup script. Keep CPU and GPU environments separate.
