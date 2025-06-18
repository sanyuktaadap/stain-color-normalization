from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from utils import colorize_seg_map

Image.MAX_IMAGE_PIXELS = None

# Step 1: Define consistent colors for each label
label_colors = {
    0: (255, 255, 255), # Background
    1: (33, 143, 166),
    2: (210, 5, 208),
    3: (5, 208, 4),
    4: (5, 5, 5),
    5: (67, 209, 247),
    6: (67, 208, 170),
    7: (6, 4, 209),
    8: (255, 102, 0),
    9: (255, 51, 0),
    10: (255, 255, 0),
}

# Step 3: Loop through all segmentation maps
masks_list = os.listdir("data/for_normalization/comparison_Images")
masks_list = [os.path.splitext(m)[0] + "_mask.png" for m in masks_list]
print(masks_list)

seg_type = "Image_Maps"
seg_paths = [os.path.join(f"data/for_normalization/{seg_type}/", m) for m in masks_list]
print(seg_paths)

for path in seg_paths:
    print(path)
    # Load segmentation map with PIL
    seg = Image.open(path).convert('L')  # 'L' mode loads as grayscale
    seg = np.array(seg)

    # Ensure it's integer labels
    seg = seg.astype(np.uint8)

    # Convert to RGB color
    colored = colorize_seg_map(seg, label_colors)

    # Show or save
    plt.figure()
    plt.imshow(colored)
    plt.title(os.path.basename(path), fontsize=17)
    plt.axis('off')
    # plt.show()

    # Save
    plt.imsave(f"data/for_normalization/colored_{seg_type}/new_{os.path.basename(path)}", colored)
