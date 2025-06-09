import numpy as np
import cv2
import glob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# Load segmentation maps (assuming each pixel value represents a class label)
seg_paths = glob.glob("data/for_normalization/KM_Masks/*.png")

# Initialize dictionary to hold image presence count for each class
class_image_presence = Counter()

for seg_path in tqdm(seg_paths):
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    unique_classes = np.unique(seg)
    for cls in unique_classes:
        class_image_presence[cls] += 1  # this image contains this class

# Plot
plt.bar(class_image_presence.keys(), class_image_presence.values())
plt.xlabel("Class Label")
plt.ylabel("Number of Images Containing the Class")
plt.title("Presence of Each Class Across Images")
plt.xticks(list(class_image_presence.keys()))  # optional, for clarity
plt.savefig("results/plots/class_stats_KM_Masks.png")