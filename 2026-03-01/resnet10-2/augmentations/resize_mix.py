import torch
import torch.nn.functional as F

def apply_resizemix(images, labels, alpha=1.0):
    B, C, H, W = images.shape
    device = images.device

    # 1. Sample lambda and permute targets
    dist = torch.distributions.beta.Beta(alpha, alpha)
    lam_array = dist.sample((B,)).to(device)

    rand_index = torch.randperm(B, device=device)
    target_a = labels
    target_b = labels[rand_index]

    # 2. Calculate initial CutMix bounding boxes
    cut_rats = torch.sqrt(1. - lam_array)
    cut_w = (W * cut_rats).long()
    cut_h = (H * cut_rats).long()

    cx = torch.randint(0, W, (B,), device=device)
    cy = torch.randint(0, H, (B,), device=device)

    # Clamped discrete coordinates
    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)

    # 3. Formalize actual clamped dimensions for continuous mapping
    actual_w = (bbx2 - bbx1).float()
    actual_h = (bby2 - bby1).float()
    actual_cx = bbx1.float() + actual_w / 2.0
    actual_cy = bby1.float() + actual_h / 2.0

    # Normalize coordinates to [-1, 1] bounds expected by grid_sample
    cx_norm = 2.0 * actual_cx / W - 1.0
    cy_norm = 2.0 * actual_cy / H - 1.0
    w_norm = actual_w / W
    h_norm = actual_h / H

    # Epsilon to prevent division by zero if a bounding box area hits 0
    eps = 1e-6
    w_norm = torch.clamp(w_norm, min=eps)
    h_norm = torch.clamp(h_norm, min=eps)

    # 4. Construct the Affine Transformation Matrix (B, 2, 3)
    theta = torch.zeros((B, 2, 3), device=device)
    theta[:, 0, 0] = 1.0 / w_norm
    theta[:, 0, 2] = -cx_norm / w_norm
    theta[:, 1, 1] = 1.0 / h_norm
    theta[:, 1, 2] = -cy_norm / h_norm

    # 5. Generate spatial grid and warp the source images
    grid = F.affine_grid(theta, size=images.shape, align_corners=False)

    # Warped tensor: Source images crushed into the bounding box, 0s everywhere else
    src_warped = F.grid_sample(
        images[rand_index],
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # 6. Generate the exact discrete binary mask
    y_grid = torch.arange(H, device=device).view(1, H, 1)
    x_grid = torch.arange(W, device=device).view(1, 1, W)
    mask_y = (y_grid >= bby1.view(B, 1, 1)) & (y_grid < bby2.view(B, 1, 1))
    mask_x = (x_grid >= bbx1.view(B, 1, 1)) & (x_grid < bbx2.view(B, 1, 1))
    mask = (mask_y & mask_x).unsqueeze(1).float()

    # 7. Blend the background and the warped foreground
    mixed_images = images * (1.0 - mask) + src_warped * mask

    # Recalculate lambda strictly based on actual pixel area modified
    actual_lam = 1.0 - (actual_w * actual_h) / (W * H)

    return mixed_images, target_a, target_b, actual_lam, mask, rand_index, theta