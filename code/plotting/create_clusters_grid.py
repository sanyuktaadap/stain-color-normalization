import glob
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec

# Plottting clusters
clusters = glob.glob("results/clustering/clusters/*")
num_clusters = len(clusters)
images_per_cluster = 6  # Reduce the number of images per cluster
aspect_ratio = 1  # Adjust for image aspect ratio (width/height)
fig_width = 16
fig_height = 16

fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(images_per_cluster, num_clusters)
plt.suptitle("Representative Images from Each Cluster", fontsize=17, y=1)

for i, cluster_path in enumerate(clusters):
    cluster_name = "Cluster: " + os.path.basename(cluster_path)
    cluster_images = glob.glob(cluster_path + "/*.png")
    random.shuffle(cluster_images)
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
plt.savefig("results/plots/clusters_grid/clusters_grid_rand7.png", dpi=300, bbox_inches='tight')