# ImageNet Classification on the MLA-100

This repository is meant to track snapshots in developments for developing an ImageNet Classifier for the Mobilint MLA100 NPU chip.

[This companion repository](https://github.com/bencejdanko/compiler-evaluation-server) is meant to be used compilations over `curl`.

### Summaries

| Model | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 2026-02-05-experimental | 5.00% | 5.00% |
| 2026-02-13-experimental | 33.23% | 29.30% |
| 2026-02-17-experimental-resnet-maps<sup>1</sup> | 63.7% | 51.30% | 
| 2026-02-20-experimental-ssl1 | 37.33% | 34.00% |

[1] Used preprocessed DINO2 GradCam annotations

![alt text](image.png)

Example output for `2026-02-20-experimental-ssl1`, experimenting with masking and encoder/decoder

## References
- [1] A ConvNet for the 2020s, https://arxiv.org/abs/2201.03545
- [2] Searching for MobileNetV3, https://arxiv.org/abs/1905.02244
- [3] EfficientNetV2, https://arxiv.org/abs/2104.00298
- [4] FasterNet, https://arxiv.org/abs/2303.03667
- [5] ConvMAE, https://arxiv.org/abs/2205.03892
- [6] Convolutional Masked Image Modeling, https://openaccess.thecvf.com/content/WACV2024/papers/Yang_Convolutional_Masked_Image_Modeling_for_Dense_Prediction_Tasks_on_Pathology_WACV_2024_paper.pdf
- [7] masked autoencoders (MAE), https://arxiv.org/abs/2111.06377
- [8] Multi-level Optimized Mask Autoencoder (MLO-MAE), https://arxiv.org/abs/2402.18128



## Contents

```
imagenet-classification-mobilint-mla-100
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
# Add reporting helper and better evaluation helpers.
└──── imagenet_mla100_02.ipynb
└──── imagenet_mla100_02.py
# A sample reporting PDF. Shows that model mainly collapses to 
# predict everything as the 2nd class (Roseate spoonbill)
└──── report_epoch_003.pdf
# introducing the 2026-02-13-experimental model,
# based off of techniques in papers [1, 2, 3, 4]
└──── imagenet_mla100_03.ipynb
└──── imagenet_mla100_03.py
|
└── 2026-02-17/
# Utilize ResNet50 to create saliency maps (15x15).
# Add a squeeze and excite blocks to model.
└──── imagenet_mla100_resnet_maps.ipynb
└──── imagenet_mla100_resnet_maps.py
|
└── 2026-02-20/
# Rollback and attempt an encoder-decoder 
# architecture as alternative to DINO2
# preprocessing
└──── imagenet_mla100_ssl1.ipynb
└──── imagenet_mla100_ssl2.py
```

