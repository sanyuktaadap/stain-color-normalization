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


def get_wsi_features_all_patches(patient_id, n_patches, model, patch_size, slide_dir, ds_arr, preprocess, device):
    """
    Extracts features from all patches of a given WSI using the specified model.

    Parameters:
    - patient_id: str, ID of the patient (corresponding to the slide file name).
    - n_patches: int, number of patches to process.
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

    slide = Image.open(os.path.join(slide_dir, f'{patient_id}.jpg'))

    for patch_idx in tqdm(range(n_patches)):
        # Reading patch coords as i=h, j=w
        i, j = ds_arr[patch_idx]
        patches_list.append(f'{patient_id}_{i}_{j}')
        # PIL Image expects width first and height later
        patch = slide.crop((j, i, j + patch_size, i + patch_size)).convert('RGB')
        patch = preprocess(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        with torch.no_grad():
            feature = model(patch)
        feature_vector = torch.mean(feature, dim=[2, 3])  # Global average pooling
        pooled_featuremap = feature_vector.squeeze(0).detach().cpu()

        combine_features.append(pooled_featuremap)

    combine_features_np = np.array([feature.cpu().numpy() for feature in combine_features])
    combine_stack = np.vstack(combine_features_np)
    return combine_stack, patches_list

# Dimensionality Reduction
def dimensionality_reduction(features, n_components=10):
    print(f'Original Feature Shape: {features.shape}')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(scaled_features)

    print(f'Reduced Feature Shape: {reduced_features.shape}')
    return reduced_features