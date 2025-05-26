from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

Image.MAX_IMAGE_PIXELS = None

# Step 1: Define consistent colors for each label
label_colors = {
    0: (255, 255, 255),  # White
    1: (33, 143, 166),   # Teal
    2: (210, 5, 208),    # Magenta
    3: (5, 208, 4),      # Green
    4: (5, 5, 5),        # Dark gray
    5: (67, 209, 247),   # Light blue
    6: (67, 208, 170),   # Aqua
    7: (6, 4, 209,),     # Indigo
    8: (255, 102, 0),    # Orange
    9: (255, 51, 0),     # Red
    10: (255, 255, 255)  # Black
}

# Step 2: Function to convert a label map to an RGB image
def colorize_seg_map(seg_map):
    h, w = seg_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)  # use uint8 for image display

    for label, color in label_colors.items():
        rgb_img[seg_map == label] = color

    return rgb_img

# Step 3: Loop through all segmentation maps
maps = os.listdir("data/for_normalization/colored_Image_Maps/")
km_path = "data/for_normalization/KM_Masks/"
# seg_paths = glob.glob("data/for_normalization/KM_Masks/*.png")
# random.shuffle(seg_paths)
seg_paths = [os.path.join(km_path, m) for m in maps]
print(seg_paths)

for path in seg_paths[:5]:
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

    # To save:
    plt.imsave(f"data/for_normalization/colored_KM_Masks/{os.path.basename(path)}", colored)
