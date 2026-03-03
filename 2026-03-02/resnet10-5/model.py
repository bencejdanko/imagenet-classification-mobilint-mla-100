import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet10(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet10, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 1, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_cam=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        fmap = self.layer4(out)
        out = self.avgpool(fmap)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        if return_cam:
            # Reconstruct CAM using the fc weights
            # fmap: (B, 512, H, W), fc.weight: (num_classes, 512)
            cam = F.conv2d(fmap, self.fc.weight.view(self.fc.out_features, self.fc.in_features, 1, 1))
            if self.fc.bias is not None:
                cam = cam + self.fc.bias.view(1, -1, 1, 1)
            return out, cam
        return out

    def generate_gradcam(self, input_batch, target_class=None):
        """
        Generates Grad-CAM masks for a given input batch.
        Targeting the last convolutional layer.
        """
        self.eval()
        
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def save_activation(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        # The last conv layer in ResNet10 is the conv2 of the last BasicBlock in layer4
        target_layer = self.layer4[-1].conv2
        hook = target_layer.register_forward_hook(save_activation)

        with torch.enable_grad():
            output = self(input_batch)
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            self.zero_grad()
            
            one_hot = torch.zeros_like(output)
            one_hot[torch.arange(output.shape[0]), target_class] = 1
            
            output.backward(gradient=one_hot, retain_graph=True)

        hook.remove()

        grads = gradients[0]
        fmaps = activations[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * fmaps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_batch.shape[2:], mode='bilinear', align_corners=False)

        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()