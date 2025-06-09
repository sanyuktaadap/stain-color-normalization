import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Local Imports
from utils import save_file, load_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from WSI patches.')
    # Patching arguments
    parser.add_argument('--images_folder', type=str, default="data/for_normalization/Images", help='Directory where the slide images are located.')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the patch to be extracted.')
    # Feature extraction arguments
    parser.add_argument('--csv_path', type=str, default="data/for_normalization/slides_list.csv", help='Path to the CSV file containing slide IDs.')
    parser.add_argument('--feat_dir', type=str, default="data/for_normalization/features", help='Directory for saving the extracted features.')
    parser.add_argument('--clust_dir', type=str, default='data/for_normalization/clustering_results')
    parser.add_argument('--clusters', type=str, default="results/clustering/clusters")

    args = parser.parse_args()

    images_folder = args.images_folder
    patch_size = args.patch_size
    csv_path = args.csv_path
    feat_dir = args.feat_dir
    clust_dir = args.clust_dir
    clusters = args.clusters

    Image.MAX_IMAGE_PIXELS = None

    image_id = load_file(csv_path)['slide_id'].tolist()

    labels_file = os.path.join(clust_dir, "labels.pkl")

    labels = load_file(labels_file)

    patches_till_now = 0

    selected_images = ["266289976.jpg", "266291365.jpg", "268005646.jpg", "286689289.jpg", "286689377.jpg", "292343407.jpg", "294219481.jpg", "310443139.jpg", "311175988.jpg"]

    for patient_id in tqdm(image_id):
        name = patient_id.split(".")[0]
        patches_path_file = os.path.join(feat_dir,
                                        name,
                                        f"{name}_VGG16_{patch_size}_patches_path.pkl")

        patches = load_file(patches_path_file)

        if patient_id not in selected_images:
            patches_till_now += len(patches)
            continue

        print(patient_id)
        image = Image.open(os.path.join(images_folder, patient_id))

        for patch in tqdm(patches):
            patch_label = labels[patches_till_now]

            clust_label = os.path.join(clusters, str(patch_label))

            os.makedirs(f"{clust_label}", exist_ok=True)

            patch_lis = patch.split("_")
            i, j = int(patch_lis[1]), int(patch_lis[2])

            if os.path.exists(os.path.join(clust_label, f"{name}_{i}_{j}.png")):
                continue

            # PIL Image expects width first and height later
            patch = image.crop((j, i, j + patch_size, i + patch_size))

            save_file(patch,
                    os.path.join(clust_label, f"{name}_{i}_{j}"),
                    ".png")

            patches_till_now += 1


