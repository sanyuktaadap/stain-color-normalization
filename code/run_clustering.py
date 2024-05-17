from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pickle

Image.MAX_IMAGE_PIXELS = None

images_path = "./data/for_normalization/Images"
# Define the file path to save the labels
labels_path = "./data/for_normalization/clustering_results/labels_dict.pkl"
# Created plots path
plot_path = "./data/for_normalization/clustering_results/plots.png"
clusters_range = range(3,14)

images_list = os.listdir(images_path)
total_images = len(images_list) - 1

full_list = []

for i, image_name in enumerate(images_list):
    print(f"Processing Image {i}/{total_images}")
    image = Image.open(os.path.join(images_path, image_name))
    w, h = image.size
    new_w = w // 2
    new_h = h // 2
    image_10x = image.resize((new_w, new_h), Image.LANCZOS)
    image_array = np.array(image_10x)
    r = image_array[0].reshape(-1,1)
    g = image_array[1].reshape(-1,1)
    b = image_array[2].reshape(-1,1)
    rgb2d = [r, g ,b]
    full_list.append(np.hstack(rgb2d))

print(f"Stacking Images")
combined_stack = np.vstack(full_list)
print(f"All images stacked")

clusters = list(clusters_range)

# Elbow Method
inertia = []
# Silhouette Score
silhouette_scores = []
# Cluster Labels
labels_dict = {}

for k in clusters:
    print(f"Starting Clustering for {k} clusters")
    kmeans = KMeans(n_clusters=k).fit(combined_stack)
    print(f"KMeans for {k} clusters: Done\nAppending inertia")
    inertia.append(kmeans.inertia_)
    print(f"Collecting labels")
    labels = kmeans.labels_
    print("Storing Labels")
    # labels_list.append(labels)
    labels_dict[k] = labels.tolist()
    print("Calculating Silhouette Score")
    silhouette_scores.append(silhouette_score(combined_stack, labels))
    print("-------------------------------")

print(f"Saving the Labels to {labels_path}")
with open(labels_path, "wb") as f:
    pickle.dump(labels_dict, f)

# # to load the labels data:
# import pickle

# # Define the file path from where to load the dictionary
# file_path = "labels_dict.pkl"

# # Load the labels_dict from the file
# with open(file_path, "rb") as f:
#     loaded_labels_dict = pickle.load(f)

print("Determining the clustering with the highest silhouette score")
best_k_index = np.argmax(silhouette_scores)
best_labels = labels_dict[clusters[best_k_index]]

print("Calculating Adjusted Rand Index (ARI) for each k against the best_labels")
ari_scores = [adjusted_rand_score(best_labels, labels_dict[k]) for k in clusters]

# Plotting the results
plt.figure(figsize=(18, 5))

# Elbow Method plot
plt.subplot(1, 3, 1)
plt.plot(clusters, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

# Silhouette Score plot
plt.subplot(1, 3, 2)
plt.plot(clusters, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')

# ARI Score plot
plt.subplot(1, 3, 3)
plt.plot(clusters, ari_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted Rand Index')
plt.title('Adjusted Rand Index For Different k')

plt.tight_layout()
plt.savefig(plot_path)
print(f"Plots saved to {plot_path}")
# plt.show()

# Print the results
best_k_by_silhouette = clusters[best_k_index]
best_silhouette_score = silhouette_scores[best_k_index]
best_ari_score = ari_scores[best_k_index]

print(f'Best k by Silhouette Score: {best_k_by_silhouette}')
print(f'Best Silhouette Score: {best_silhouette_score}')
print(f'Best ARI Score: {best_ari_score}')