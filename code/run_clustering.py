from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

Image.MAX_IMAGE_PIXELS = None

def run_clustering(obj, n_clust):
    print(f"Starting Clustering for {n_clust} clusters")
    kmeans = KMeans(n_clusters=n_clust).fit(obj)
    print(f"KMeans Done \nCollecting labels")
    labels = kmeans.labels_
    return labels