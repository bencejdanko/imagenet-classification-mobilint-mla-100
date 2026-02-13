# ImageNet Classification on the MLA-100

This repository is meant to track snapshots in developments for developing an ImageNet Classifier for the Mobilint MLA100 NPU chip.

[This companion repository](https://github.com/bencejdanko/compiler-evaluation-server) is meant to be used compilations over `curl`.

```
├──
└── 2026-02-05/
# Prepared baseline data preparation, dataloader, DNN model, and export
├──── imagenet_mla100.ipynb
├──── imagenet_mla100.py
# Prepare report document frame
└──── imagenet_mla.md
|
└── 2026-02-10/
# Prepared an inference script on the MLA-100 device.
├──── inference_script.py
# Reported issues with inference.
└──── issues.md
|
└── 2026-02-13/
# Compilable model that states `Output Shape      = [1, 1, 20]`.
# More explicit shape and output for MXQ inference.
├──── imagenet_mla100.ipynb
├──── imagenet_mla100.py
# Reported issues with inference.
└──── issues.md

```