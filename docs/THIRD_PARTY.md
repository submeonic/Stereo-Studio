# Third-party dependencies

This project uses Pillow, NumPy, OpenCV contrib, PySide6/Qt, and FFmpeg under their respective upstream licenses. They are installed dependencies, not copied source in this repository.

## Optional NVIDIA backend

- Official source: https://github.com/NVlabs/Fast-FoundationStereo
- Tested upstream revision: `476f4249561f7c79ca707326954f9255643412a6`
- Checkpoint: official research `23-36-37/model_best_bp2_serialize.pth`, linked from the upstream README's Google Drive folder.
- SHA-256: `af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990`
- Tested runtime: PyTorch 2.6.0 + CUDA 12.4, torchvision 0.21.0; full versions in `requirements-gpu.lock`.

Upstream code remains in the ignored `third_party/Fast-FoundationStereo` checkout, with its own license intact. Weights remain in ignored `models/`. Neither is included in this application's Git commit. The adapter loads the explicitly installed serialized model; user photo inputs are never unpickled. Only use checkpoints from a trusted source.

The source and research weights have their own NVIDIA terms. Upstream distinguishes the research checkpoints from its separately licensed commercial model and TAO code. This application does not relicense the upstream model or assert commercial-use rights. Consult the upstream license and model documentation before redistribution or commercial deployment.

The authors also offer a larger FoundationStereo model aimed at offline accuracy. This release implements and tests Fast-FoundationStereo; it does not claim to be the best possible model on every photograph.
