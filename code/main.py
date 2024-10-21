import argparse
import os
from PIL import Image
import pandas as pd
import h5py
import numpy as np
import pickle
from tqdm import tqdm
import sys

# Torch imports
import torch
from torchvision import transforms as T
from torchvision import models

# Local Imports
from extract_patches import extract_patches
from extract_features import get_wsi_features_all_patches, dimensionality_reduction
from run_clustering import run_clustering
from utils import save_file, load_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from WSI patches.')
    # Patching arguments
    parser.add_argument('--images_folder', type=str, default="data/for_normalization/Images", help='Directory where the slide images are located.')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the patch to be extracted.')
    parser.add_argument('--hdf5_folder', type=str, default="data/for_normalization/patches", help='Directory to save the extracted patches and coordinates.')
    # Feature extraction arguments
    parser.add_argument('--csv_path', type=str, default="data/for_normalization/slides_list.csv", help='Path to the CSV file containing slide IDs.')
    parser.add_argument('--feat_dir', type=str, default="data/for_normalization/features", help='Directory for saving the extracted features.')
    parser.add_argument('--clust_dir', type=str, default='data/for_normalization/clustering_results')
    parser.add_argument('--n_comp', type=int, default=32, help='Number of components for PCA.')
    parser.add_argument('--n_clust', type=int, default=7, help='Number of clusters for K-Means')
    parser.add_argument('--mask_fold',  type=str, default='data/for_normalization/KM_Masks')

    args = parser.parse_args()

    images_folder = args.images_folder
    patch_size = args.patch_size
    hdf5_folder = args.hdf5_folder
    csv_path = args.csv_path
    feat_dir = args.feat_dir
    clust_dir = args.clust_dir
    n_comp = args.n_comp
    n_clust = args.n_clust
    mask_fold = args.mask_fold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Image.MAX_IMAGE_PIXELS = None

    image_id = load_file(csv_path)['slide_id'].tolist()
    total_images = len(image_id)

    # Step 1: Extract patches with coordinates
    images = os.listdir(images_folder)
    print(f"------- Patching {len(images)} slides -------")
    extract_patches(images_folder, patch_size, hdf5_folder)
    print(f"Patches saved in {hdf5_folder}\n\n")

    # Step 2: Extract Features
    # Load the pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    model = torch.nn.Sequential(*list(model.features.children())[:-1])
    model.eval()

    total_images = len(image_id)

    print(f"------- Starting Feature Extraction -------")

    for i, patient_id in enumerate(image_id[::-1]):
        name = patient_id.split(".")[0]
        path = os.path.join(feat_dir, name)
        print(f"{i+1}/{total_images} - {name}")
        if os.path.exists(path):
            continue

        patient_id = patient_id.split(".")[0]
        pt_hd5 = f"{os.path.join(hdf5_folder, patient_id)}.h5"
        attr_dict = {}
        with h5py.File(f'{pt_hd5}', "r") as f:
            a_group_key = list(f.keys())[0]
            ds_arr = f[a_group_key][()]
            for k, v in f[a_group_key].attrs.items():
                attr_dict[k] = v

        total_patches = len(ds_arr)
        print(f'Total patches: {total_patches}')

        preprocess = T.Compose([
            T.Resize(size=256),
            T.ToTensor(),
        ])

        wsi_featuremap, patches_list = get_wsi_features_all_patches(patient_id, total_patches, model, patch_size, images_folder, ds_arr, preprocess, device)
        output_dir = os.path.join(feat_dir, patient_id)
        os.makedirs(output_dir, exist_ok=True)

        save_file(wsi_featuremap,
                  os.path.join(output_dir, f'{patient_id}_VGG16_256'),
                  ".npy")
        save_file(patches_list,
                  os.path.join(output_dir, f'{patient_id}_VGG16_256_patches_path'),
                  ".pkl")

        print(f"Features saved in {output_dir}")

    # Step 3: Concatenate all wsi_featuremaps
    combined_list = []
    for i, patient_id in tqdm(enumerate(image_id)):
        name = patient_id.split(".")[0]
        print(f"{i}/{total_images-1} - {name}")
        output_dir = os.path.join(feat_dir, name)

        # Load wsi_featuremap from the .npy file
        wsi_featuremap = load_file(os.path.join(output_dir,f'{name}_VGG16_256.npy'))
        combined_list.append(wsi_featuremap)

    combined_features = np.concatenate(combined_list, axis=0)
    save_file(combined_features,
              os.path.join(feat_dir, f'combined_feature_maps'),
              ".npy")

    print(f"Concatenated Array Shape: {combined_features.shape}")

    # Step 4: Reduce Extracted Features
    print('------- Dimensionality Reduction -------')
    if not os.path.exists(os.path.join(feat_dir, f'combined_feature_maps.npy')):
        print("Feature maps not concatenated!! Exiting...")
        sys.exit(1)

    # Load wsi_featuremap from the .npy file
    combined_features = load_file(os.path.join(feat_dir, f'combined_feature_maps.npy'))

    reduced_features = dimensionality_reduction(combined_features, n_comp)

    save_file(reduced_features,
              os.path.join(feat_dir, f'combined_VGG16_256_{n_comp}_reduced_features'),
              ".npy")

    print(f'Dimensionality Reduction is Done. Results Saved in {feat_dir}. Components: {n_comp}')

    # Step 5: Run K-Means clustering
    print("\n------- Clustering -------")
    reduced_features = load_file(os.path.join(feat_dir, f'combined_VGG16_256_{n_comp}_reduced_features.npy'))

    labels = run_clustering(reduced_features, n_clust)

    labels = labels.tolist()
    save_file(labels,
              os.path.join(clust_dir, "labels"),
              ".pkl")

    print(f'Total Labels: {len(labels)}')

    # Step 6: Create Label Maps
    print("------- Creating Label Maps -------")

    os.makedirs(mask_fold, exist_ok=True)

    labels_file = os.path.join(clust_dir, "labels.pkl")

    labels = load_file(labels_file)

    patches_till_now = 0
    labels_dict = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7}

    for i, patient_id in tqdm(enumerate(image_id)):
        image = Image.open(os.path.join(images_folder, patient_id))
        h, w = image.size
        empty_mask = np.zeros((w, h), dtype=np.uint8)

        name = patient_id.split(".")[0]
        print(f"{i+1}/{total_images} - {name}")
        patches_path_file = os.path.join(feat_dir,
                                         name,
                                         f"{name}_VGG16_{patch_size}_patches_path.pkl")
        patches = load_file(patches_path_file)

        for patch in patches:
            patch_label = labels[patches_till_now]

            patch_lis = patch.split("_")
            x, y = int(patch_lis[1]), int(patch_lis[2])
            patch = image.crop((x, y, x+patch_size, y+patch_size))

            empty_mask[x: x + patch_size, y: y + patch_size] = labels_dict[patch_label]

            patches_till_now += 1

        # Convert the mask to an image and save it as PNG
        mask_image = Image.fromarray(empty_mask)
        mask_image.save()
        save_file(mask_image,
                  os.path.join(mask_fold, f"{name}_mask"),
                  ".png")

    print(f"-------Masks saved at {mask_fold}-------")