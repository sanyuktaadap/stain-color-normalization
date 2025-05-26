import os
import numpy as np
import cv2
import glob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import matplotlib.gridspec as gridspec

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

# Plottting clusters
clusters = glob.glob("results/clustering/cluster_plots/*")
num_clusters = len(clusters)
images_per_cluster = 6  # Reduce the number of images per cluster
aspect_ratio = 1  # Adjust for image aspect ratio (width/height)
fig_width = 16
fig_height = 16

fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(images_per_cluster, num_clusters)
plt.suptitle("Representative Images from Each Cluster", fontsize=22, y=1)

for i, cluster_path in enumerate(clusters):
    cluster_name = "Cluster: " + os.path.basename(cluster_path)
    cluster_images = glob.glob(cluster_path + "/*.png")
    random.shuffle(cluster_images)
    representative_images = cluster_images[:images_per_cluster]

    # Plot the representative images for the current cluster
    for j, img_path in enumerate(representative_images):
        ax = plt.subplot(gs[j, i])  # Start row index from 0
        img = plt.imread(img_path)
        ax.imshow(img, aspect='equal')
        ax.axis('off')

        # Add cluster name only above the first image in the column
        if j == 0:
            ax.set_title(cluster_name, fontsize=17, pad=5)

plt.tight_layout()
plt.savefig("results/plots/clusters_grid.png", dpi=300, bbox_inches='tight')