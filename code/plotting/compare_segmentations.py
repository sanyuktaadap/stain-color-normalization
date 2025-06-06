import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

Image.MAX_IMAGE_PIXELS = None

# Resize helper
def resize_for_display(pil_img, max_size=(224, 224)):
    img = pil_img.copy()
    img.thumbnail(max_size, Image.ANTIALIAS)
    return img

# Smoothen image helper
def smooth_mask(pil_img, size=3):
    """Applies median filter to smoothen mask edges without blurring labels."""
    np_img = np.array(pil_img)
    if np_img.ndim == 3:
        # Apply median filter to each channel separately
        smoothed = np.stack([median_filter(np_img[:, :, c], size=size) for c in range(3)], axis=-1)
    else:
        smoothed = median_filter(np_img, size=size)
    return Image.fromarray(smoothed.astype(np.uint8))

# Folder paths
img_folder = "data/for_normalization/comparison_Images"
gt_mask_folder = "data/for_normalization/colored_Image_Maps"
pred_mask_folder = "data/for_normalization/colored_KM_Masks"

# Get sorted list of image filenames
image_filenames = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
print(f"Filenames: {image_filenames}")

n_rows = len(image_filenames)
n_cols = 4

# Set up the figure
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 4 * n_rows))

column_titles = ["Original Image", "Original Mask", "Our Mask", "Our Smoothed Mask"]

# Add column titles
for j in range(n_cols):
    axs[0][j].set_title(column_titles[j], fontsize=17)

# Plot images
kernel_sizes = [3, 5, 7, 9]
for size in kernel_sizes:
    for i, filename in enumerate(image_filenames):
        name_prefix = os.path.splitext(filename)[0]
        original_path = os.path.join(img_folder, filename)
        gt_mask_path = os.path.join(gt_mask_folder, f"{name_prefix}_mask.png")
        pred_mask_path = os.path.join(pred_mask_folder, f"{name_prefix}_mask.png")

        if not all([os.path.exists(p) for p in [original_path, gt_mask_path, pred_mask_path]]):
            print(f"Skipping {filename} due to missing file")
            continue

        # Load images using PIL
        original_img = Image.open(original_path)
        gt_mask_img = Image.open(gt_mask_path)
        pred_mask_img = Image.open(pred_mask_path)

        original_img = resize_for_display(original_img)
        gt_mask_img = resize_for_display(gt_mask_img)
        pred_mask_img = resize_for_display(pred_mask_img)

        # Smoothen the predicted mask
        smoothed_pred_img = smooth_mask(pred_mask_img, size=size)
        smoothed_pred_img = resize_for_display(smoothed_pred_img)

        print(f"{filename}: original_img size = {original_img.size}")

        # Plot each in its respective column
        axs[i][0].imshow(original_img)
        axs[i][1].imshow(gt_mask_img)
        axs[i][2].imshow(pred_mask_img)
        axs[i][3].imshow(smoothed_pred_img)

        # Remove axes
        for j in range(n_cols):
            axs[i][j].axis('off')

    # Adjust layout and display
    fig.suptitle("Segmentation Comparison", fontsize=22, y=1)
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.savefig(f"results/plots/seg_comparison_kernel{size}.png", dpi=300)