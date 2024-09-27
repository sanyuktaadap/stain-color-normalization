# -*- coding: utf-8 -*-

"""
Created on Sun Oct 4 10:37:51 2022

@author: shubham
"""

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle
import h5py
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Torch imports
import torch
from torchvision import transforms as T
from torchvision import models


def get_wsi_features_all_patches(patient_id, n, model, patch_size, slide_dir, ds_arr, preprocess, device):
    """
    Extracts features from all patches of a given WSI using the specified model.

    Parameters:
    - patient_id: str, ID of the patient (corresponding to the slide file name).
    - n: int, number of patches to process.
    - model: pre-trained model for feature extraction.
    - patch_size: int, Patch size.
    - slide_dir: str, directory where the slide images are located.
    - preprocess: transforms applied to the image

    Returns:
    - combine_features_np: numpy array, extracted features.
    - patches_list: list of str, list of patch identifiers.
    """
    combine_features = []
    patches_list = []

    model.to(device)

    for i in tqdm(range(n)):
        slide = Image.open(os.path.join(slide_dir, f'{patient_id}.jpg'))
        x, y = ds_arr[i]
        patches_list.append(f'{patient_id}_{x}_{y}')
        patch = slide.crop((x, y, x + patch_size, y + patch_size)).convert('RGB')
        patch = preprocess(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        with torch.no_grad():
            feature = model(patch)

        feature_vector = torch.mean(feature, dim=[2, 3])  # Global average pooling
        pooled_featuremap = feature_vector.squeeze(0)

        combine_features.append(pooled_featuremap)
        slide.close()

    combine_features_np = np.array([feature.cpu().numpy() for feature in combine_features])
    combine_stack = np.vstack(combine_features_np)
    return combine_stack, patches_list

# Dimensionality Reduction
def dimensionality_reduction(features, n_components=10):
    print('Dimensionality Reduction...')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(scaled_features)

    print(f'Dimensionality Reduction is Done. Components: {n_components}')
    print(reduced_features.shape)
    return reduced_features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from WSI patches.')
    parser.add_argument('--slide_dir', type=str, required=True, help='Directory where the slide images are located.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing slide IDs.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for saving the extracted features.')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for saving the extracted features.')
    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = None

    image_id = pd.read_csv(args.csv_path)['slide_id'].to_list()

    # Load the pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Step 1: Modify the model to remove the last pooling layer
    model = torch.nn.Sequential(*list(model.features.children())[:-1])
    model.eval()

    patch_size = 256

    for patient_id in tqdm(image_id):
        foldername = os.path.join(args.slide_dir, patient_id)
        folder_name = os.path.basename(foldername)
        print(folder_name)
        attr_dict = {}
        with h5py.File(f'{foldername}.h5', "r") as f:
            a_group_key = list(f.keys())[0]
            ds_arr = f[a_group_key][()]
            for k, v in f[a_group_key].attrs.items():
                attr_dict[k] = v

        total_patches = len(ds_arr)
        print(f'Total patches: {total_patches}')

        preprocess = T.Compose([
            T.ToPILImage()(),
            T.Resize(size=256),
            T.ToTensor(),
        ])

        wsi_featuremap, patches_list = get_wsi_features_all_patches(folder_name, total_patches, fe, patch_size, args.slide_dir, ds_arr, preprocess)
        output_dir = os.path.join(args.root_dir, args.output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, f'{folder_name}_VGG16_256.npy'), wsi_featuremap, allow_pickle=True)
        with open(os.path.join(output_dir, f'{folder_name}_VGG16_256_patches_list.pkl'), 'wb') as f:
            pickle.dump(patches_list, f)
