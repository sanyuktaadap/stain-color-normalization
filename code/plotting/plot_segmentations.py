from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

Image.MAX_IMAGE_PIXELS = None

# Step 1: Define consistent colors for each label
label_colors = {
    0: (245, 245, 245),  # Soft White (light gray-white)
    1: (255, 102, 102),  # Soft Red (pastel red)
    2: (255, 153, 255),  # Soft Magenta (light pink-magenta)
    3: (102, 204, 204),  # Soft Teal
    4: (255, 255, 153),  # Soft Yellow (light yellow)
    5: (153, 204, 255),  # Soft Blue (light sky blue)
    6: (64, 64, 64),     # Soft Black (dark gray)
    7: (204, 153, 255),  # Soft Purple (lavender)
    8: (255, 204, 153),  # Soft Orange (peachy)
    9: (153, 255, 153),  # Soft Green (mint green)
    10: (204, 204, 153), # Soft Olive (muted khaki)
}


# Step 2: Function to convert a label map to an RGB image
def colorize_seg_map(seg_map):
    h, w = seg_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)  # use uint8 for image display

    for label, color in label_colors.items():
        rgb_img[seg_map == label] = color

    return rgb_img

# Step 3: Loop through all segmentation maps
masks_list = os.listdir("data/for_normalization/comparison_Images")
masks_list = [os.path.splitext(m)[0] + "_mask.png" for m in masks_list]
print(masks_list)

seg_type = "KM_Masks"
seg_paths = [os.path.join(f"data/for_normalization/{seg_type}/", m) for m in masks_list]
print(seg_paths)

for path in seg_paths:
    # Load segmentation map with PIL
    seg = Image.open(path).convert('L')  # 'L' mode loads as grayscale
    seg = np.array(seg)

    # Ensure it's integer labels
    seg = seg.astype(np.uint8)

    # Convert to RGB color
    colored = colorize_seg_map(seg)

    # Show or save
    plt.figure()
    plt.imshow(colored)
    plt.title(os.path.basename(path))
    plt.axis('off')
    # plt.show()

    # Save
    plt.imsave(f"data/for_normalization/colored_{seg_type}/{os.path.basename(path)}", colored)
