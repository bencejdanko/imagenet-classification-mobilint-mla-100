import torch
import torch.nn as nn

class SpatialAttentionGate(nn.Module):
    """Generates a spatial heatmap and multiplies it with the feature map."""
    def __init__(self, channels):
        super().__init__()
        # Mini-decoder to find the most relevant spatial features
        self.att_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.GroupNorm(4, channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid() # Forces values between 0.0 and 1.0
        )

    def forward(self, x):
        attention_map = self.att_conv(x)
        # NPU Allowed Operation: Mul
        attended_features = x * attention_map
        return attended_features, attention_map

class NPUDecoder(nn.Module):
    """Scales the 15x15 latent space back to a 240x240 image."""
    def __init__(self, in_channels=256):
        super().__init__()
        # Using Nearest Upsample + Conv to avoid checkerboard artifacts (NPU allowed)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 15 -> 30
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'), # 30 -> 60
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'), # 60 -> 120
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'), # 120 -> 240
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid() # Images are normalized 0-1, so output must be 0-1
        )

    def forward(self, x):
        return self.decoder(x)

class NPUModel(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        # --- STAGE 1: STEM ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32), # Replaced BatchNorm
            nn.ReLU(inplace=True)
        )

        # --- STAGE 2: MBCONV ---
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, 64)
        )

        # --- STAGE 3: DOWNSAMPLE ---
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True)
        )

        # --- STAGE 4: FEATURE REFINEMENT ---
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=False),
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True)
        )

        # --- THE SPLIT ---
        self.attention_gate = SpatialAttentionGate(channels=256)
        self.decoder = NPUDecoder(in_channels=256)

        # --- CLASSIFIER HEAD ---
        self.classifier_conv = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() # NPU Allowed

    def forward(self, x, return_reconstruction=False):
        # Backbone Feature Extraction
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x) # Shape: (B, 256, 15, 15)

        # 1. Attention Filtering
        attended_features, attention_map = self.attention_gate(features)

        # 2. Classification Branch (using filtered features)
        cls_out = self.classifier_conv(attended_features)
        cls_out = self.global_pool(cls_out)
        logits = self.flatten(cls_out)

        # During inference (ONNX export), we only care about the logits.
        if not return_reconstruction:
            return logits

        # 3. Reconstruction Branch (forcing the network to learn global context)
        # We decode from the *unfiltered* features so the encoder has to learn everything.
        reconstructed_img = self.decoder(features)

        return logits, reconstructed_img, attention_map