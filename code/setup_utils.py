import pandas as pd
import os
import random
import shutil
import numpy as np
from skimage.io import imread
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def get_images_from_path(source_folder, path, destination_folder):
    image_path = path.split("/", 7)[-1]
    source_image_path = os.path.join(source_folder, image_path)
    shutil.copy(source_image_path, destination_folder)

def get_maps_from_path(source_folder, path, destination_folder):
    temp_path = path.split("/", 8)[-1]
    source_map_path = os.path.join(source_folder, temp_path)
    if os.path.exists(source_map_path):
        shutil.copy(source_map_path, destination_folder)

def get_images_from_selected_maps(maps_folder, images_folder, destination_folder):
    maps = os.listdir(maps_folder)
    images = os.listdir(images_folder)
    for map in maps:
        number = map.split("_")[-1].split(".")[0]
        image_name = f"{number}.jpg"
        if image_name in images:
            shutil.copy(os.path.join(images_folder, image_name), destination_folder)

def randomly_select_maps(csv_path, maps_folder, destination_folder, num_images):
    maps_list = os.listdir(maps_folder)
    data = pd.read_csv(csv_path)
    selected_maps_paths = data.iloc[:, 0]
    selected_maps_list = []
    for path in selected_maps_paths:
        map_name = path.split("/")[-1]
        if map_name in maps_list:
            selected_maps_list.append(map_name)
    print(f"Total maps found: {len(selected_maps_list)}")
    if len(selected_maps_list) > num_images:
        random_maps = random.sample(selected_maps_list, 120)
        for map in random_maps:
            shutil.copy(os.path.join(maps_folder, map), destination_folder)
    else:
        print("Not enough maps found to copy")

def get_pi_value_count(roi_list, mask_folder_path, og_img_folder_path, norm_img_folder_path):
    all_masks = os.listdir(mask_folder_path)
    # Create a dictionary that contains all other dictionaries
    full_dict = {}
    for roi in roi_list:
        og = "Original_ROI_" + str(roi)
        norm = "Normalized_ROI_" + str(roi)
        full_dict[og] = {}
        full_dict[norm] = {}
    for key in full_dict.keys():
        # Update the value for the key with individual elements
        full_dict[key] = {}
        for i in list(range(0,256)):
            full_dict[key][i] = 0
    # Loop over each mask
    for a, mask_name in enumerate(all_masks):
        print(f"Mask {a}: {mask_name}")
        mask_path = os.path.join(mask_folder_path, mask_name)
        mask = imread(mask_path)
        # Get ROIs in the mask
        uni = np.unique(mask)
        print(f"    Total ROI in Mask {a}: {len(uni)} ({uni})")
        # Loop over each ROI
        for num in uni:
            print(f"    Processing ROI {num}")
            # Get the locations of the ROI
            indices = np.where(mask == num)
            locs = list(zip(indices[0], indices[1]))
            # Get original and normalized image names from mask name
            og_img_name = mask_name.split("_")[-1].split(".")[0] + ".jpg"
            norm_img_name = mask_name.split("_")[-1].split(".")[0] + "_Normalized.png"
            # Read Original and Normalized Image
            og_img = imread(os.path.join(og_img_folder_path, og_img_name))
            norm_img = imread(os.path.join(norm_img_folder_path, norm_img_name))
            # Loop over the locations obtained from the mask
            for loc in tqdm(locs):
                # Obtain RBG values in the location
                og_vals = og_img[loc[0],loc[1]]
                norm_vals = norm_img[loc[0],loc[1]]
                # Loop over each location
                for i in og_vals:
                    full_dict[f"Original_ROI_{num}"][i] += 1
                for j in norm_vals:
                    full_dict[f"Normalized_ROI_{num}"][j] += 1
            print(f"    ROI {num} process completed")
            print()
    return full_dict