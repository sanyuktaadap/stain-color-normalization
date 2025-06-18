import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

def colorize_seg_map(seg_map, label_colors):
    h, w = seg_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)  # use uint8 for image display

    for label, color in label_colors.items():
        rgb_img[seg_map == label] = color

    return rgb_img

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