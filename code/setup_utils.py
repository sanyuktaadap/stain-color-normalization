import pandas as pd
import os
import random
import shutil

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