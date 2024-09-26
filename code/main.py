import argparse
import os
from PIL import Image
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import h5py
from tqdm import tqdm
import numpy as np
import pickle
import torch.hub as hub
from torchvision import transforms as T
import torch

# Local Imports
from extract_patches import extract_patches
from extract_features import get_wsi_features_all_patches, dimensionality_reduction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from WSI patches.')
    # Patching arguments
    parser.add_argument('--images_folder', type=str, default="data/for_normalization/Images", help='Directory where the slide images are located.')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the patch to be extracted.')
    parser.add_argument('--hdf5_folder', type=str, default="data/for_normalization/patches", help='Directory to save the extracted patches and coordinates.')
    # Feature extraction arguments
    parser.add_argument('--csv_path', type=str, default="data/for_normalization/slides_list.csv", help='Path to the CSV file containing slide IDs.')
    parser.add_argument('--feat_dir', type=str, default="data/for_normalization/features", help='Directory for saving the extracted features.')
    parser.add_argument('--n_comp', type=int, default=3, help='Number of components for PCA.')
    args = parser.parse_args()

    images_folder = args.images_folder
    patch_size = args.patch_size
    hdf5_folder = args.hdf5_folder
    feat_dir = args.feat_dir
    n_comp = args.n_comp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Image.MAX_IMAGE_PIXELS = None

    # # Step 1: Extract patches with coordinates
    # images = os.listdir(images_folder)
    # print(f"Patching {len(images)} slides")
    # extract_patches(images_folder, patch_size, hdf5_folder)
    # print(f"Patches saved in {hdf5_folder}")

    # Step 2: Extract Features
    image_id = pd.read_csv(args.csv_path)['slide_id'].to_list()
    model = hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    fe = model.features
    total_images = len(image_id)

    for i, patient_id in enumerate(image_id[:2]):
        print(f"{i}/{total_images}")
        patient_id = patient_id.split(".")[0]
        pt_hd5 = f"{os.path.join(hdf5_folder, patient_id)}.h5"
        attr_dict = {}
        with h5py.File(f'{pt_hd5}', "r") as f:
            a_group_key = list(f.keys())[0]
            ds_arr = f[a_group_key][()]
            print(ds_arr)
            for k, v in f[a_group_key].attrs.items():
                attr_dict[k] = v

        total_patches = len(ds_arr)
        print(f'Total patches: {total_patches}')

        preprocess = T.Compose([
            T.Resize(size=224),
            T.ToTensor(),
        ])

        wsi_featuremap, patches_list = get_wsi_features_all_patches(patient_id, total_patches, model, patch_size, images_folder, ds_arr, preprocess, device)
        output_dir = os.path.join(feat_dir, patient_id)
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, f'{patient_id}_VGG16_256.npy'), wsi_featuremap, allow_pickle=True)
        with open(os.path.join(output_dir, f'{patient_id}_VGG16_256_patches_path.pkl'), 'wb') as f:
            pickle.dump(patches_list, f)

        print(f"Features saved in {feat_dir}")

        reduced_features = dimensionality_reduction(wsi_featuremap, n_comp)
        np.save(os.path.join(output_dir, f'{patient_id}_VGG16_256_reduced_features.npy'), wsi_featuremap, allow_pickle=True)