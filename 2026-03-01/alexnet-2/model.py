import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def generate_gradcam(self, input_batch, target_class=None):
        """
        Generates Grad-CAM masks for a given input batch.
        Targeting the last convolutional layer of the features block.
        """
        self.eval()
        
        # We need gradients for the activations
        # Ensure input_batch requires grad if not already (though usually it doesn't)
        # but we definitely need to enable grad for the backward pass.
        
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def save_activation(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        # The last conv layer is self.features[10]
        target_layer = self.features[10]
        hook = target_layer.register_forward_hook(save_activation)

        # Forward pass
        # Enable gradients for this block even if called under torch.no_grad()
        with torch.enable_grad():
            output = self(input_batch)
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Zero gradients
            self.zero_grad()
            
            # Create a one-hot like mask for the target classes
            one_hot = torch.zeros_like(output)
            one_hot[torch.arange(output.shape[0]), target_class] = 1
            
            # Backward pass to get gradients
            output.backward(gradient=one_hot, retain_graph=True)

        # Remove hook
        hook.remove()

        # Compute Grad-CAM
        # gradients[0] shape: [B, 256, H, W]
        # activations[0] shape: [B, 256, H, W]
        grads = gradients[0]
        fmaps = activations[0]

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        # Weighted sum of feature maps
        cam = (weights * fmaps).sum(dim=1, keepdim=True)
        # ReLU to keep only positive influence
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_batch.shape[2:], mode='bilinear', align_corners=False)

        # Normalize per image in batch
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()