import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstruction(masked, target, recon, attention, epoch):
    """Plots the first image in the batch to track learning progress."""
    # Move tensors to CPU and convert to HWC numpy arrays for plotting
    masked_img = masked[0].cpu().permute(1, 2, 0).numpy()
    target_img = target[0].cpu().permute(1, 2, 0).numpy()

    # Detach gradients before moving to CPU
    recon_img = recon[0].detach().cpu().permute(1, 2, 0).numpy()

    # Attention map is (1, H, W). We squeeze it to (H, W)
    att_map = attention[0, 0].detach().cpu().numpy()

    # Clip values to ensure they stay in the 0-1 range for matplotlib
    masked_img = np.clip(masked_img, 0, 1)
    target_img = np.clip(target_img, 0, 1)
    recon_img = np.clip(recon_img, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(target_img)
    axes[0].set_title("Original Target")
    axes[0].axis("off")

    axes[1].imshow(masked_img)
    axes[1].set_title("Masked Input")
    axes[1].axis("off")

    axes[2].imshow(recon_img)
    axes[2].set_title("Model Reconstruction")
    axes[2].axis("off")

    # Use a heatmap colormap for the attention
    im = axes[3].imshow(att_map, cmap='jet')
    axes[3].set_title("Spatial Attention Map")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Epoch {epoch} Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()
