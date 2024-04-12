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

def randomly_select_images(maps_folder, num_images, destination_folder):
    all_maps = os.listdir(maps_folder)
    random_maps = random.sample(all_maps, num_images)
    for map in random_maps:
        shutil.copy(os.path.join(maps_folder, map), destination_folder)

def get_images_from_selected_maps(maps_folder, images_folder, destination_folder):
    maps = os.listdir(maps_folder)
    images = os.listdir(images_folder)
    for map in maps:
        number = map.split("_")[-1].split(".")[0]
        image_name = f"{number}.jpg"
        if image_name in images:
            shutil.copy(os.path.join(images_folder, image_name), destination_folder)