import math

def apply_hmix(images, labels, alpha=1.0, r=0.5):
    B, C, H, W = images.shape
    device = images.device

    dist = torch.distributions.beta.Beta(alpha, alpha)
    lam_array = dist.sample((B,)).to(device)

    rand_index = torch.randperm(B, device=device)
    target_a = labels
    target_b = labels[rand_index]

    mixup_lams = lam_array / (1.0 - (1.0 - lam_array) * r)
    mix_tensor = mixup_lams.view(B, 1, 1, 1)

    base_mix = mix_tensor * images + (1.0 - mix_tensor) * images[rand_index]

    cut_rats = torch.sqrt(1. - lam_array) * math.sqrt(r)
    cut_w = (W * cut_rats).long()
    cut_h = (H * cut_rats).long()

    cx = torch.randint(0, W, (B,), device=device)
    cy = torch.randint(0, H, (B,), device=device)

    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)

    y_grid = torch.arange(H, device=device).view(1, H, 1)
    x_grid = torch.arange(W, device=device).view(1, 1, W)

    mask_y = (y_grid >= bby1.view(B, 1, 1)) & (y_grid < bby2.view(B, 1, 1))
    mask_x = (x_grid >= bbx1.view(B, 1, 1)) & (x_grid < bbx2.view(B, 1, 1))

    mask = (mask_y & mask_x).unsqueeze(1).float()

    mixed_images = base_mix * (1.0 - mask) + images[rand_index] * mask

    return mixed_images, target_a, target_b, lam_array, mask, rand_index, mix_tensor