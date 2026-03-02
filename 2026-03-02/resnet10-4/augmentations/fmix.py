import math
import torch

def apply_fmix(images, labels, alpha=1.0, decay_power=3.0):
    B, C, H, W = images.shape
    device = images.device

    dist = torch.distributions.beta.Beta(alpha, alpha)
    lam_array = dist.sample((B,)).to(device)

    rand_index = torch.randperm(B, device=device)
    target_a = labels
    target_b = labels[rand_index]

    freqs_x = torch.fft.fftfreq(W, device=device)
    freqs_y = torch.fft.fftfreq(H, device=device)
    fy, fx = torch.meshgrid(freqs_y, freqs_x, indexing='ij')

    freq_sq = fx**2 + fy**2
    freq_sq[0, 0] = 1.0

    amp = 1.0 / (freq_sq ** (decay_power / 2.0))
    amp[0, 0] = 0.0
    amp = amp.unsqueeze(0).expand(B, H, W)

    phase = torch.rand((B, H, W), device=device) * 2 * math.pi
    complex_spec = amp * torch.exp(1j * phase)

    img = torch.fft.ifft2(complex_spec).real

    img_flat = img.reshape(B, -1)

    idx = torch.clamp(((1.0 - lam_array) * (H * W)).long(), 0, H * W - 1)
    sorted_img, _ = torch.sort(img_flat, dim=1)
    thresholds = sorted_img[torch.arange(B, device=device), idx].view(B, 1, 1, 1)

    mask = (img.unsqueeze(1) > thresholds).float()

    mixed_images = mask * images + (1.0 - mask) * images[rand_index]
    actual_lam = mask.mean(dim=(1, 2, 3))

    return mixed_images, target_a, target_b, actual_lam, mask, rand_index