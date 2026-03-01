def apply_mixup(images, labels, alpha=1.0):
    B = images.size(0)
    device = images.device

    dist = torch.distributions.beta.Beta(alpha, alpha)
    lam_array = dist.sample((B,)).to(device)

    rand_index = torch.randperm(B, device=device)
    target_a = labels
    target_b = labels[rand_index]

    lam_view = lam_array.view(B, 1, 1, 1)

    mixed_images = lam_view * images + (1.0 - lam_view) * images[rand_index, :]
    mask = torch.zeros((B, 1, images.size(2), images.size(3)), device=device)

    return mixed_images, target_a, target_b, lam_array, mask, rand_index, lam_view