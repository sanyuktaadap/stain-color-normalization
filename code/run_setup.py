from setup_utils import *
import pandas as pd
import random

verified_images_csv = "./data/Csv_Files/VerifiedIvyGap2176ImageCohortListWithExclusions.csv"
source_images_folder = "./data/RAW"
images_folder = "./data/Images"
source_maps_folder = "./data/label_maps/label_maps"
maps_folder = "./data/Image_Maps"
norm_images_folder = './data/for_normalization/Images'
norm_maps_folder = "./data/for_normalization/Image_Maps"
num_images = 120
csv_path = "data/Csv_Files/VerifiedIvyGap858ImageCohort.csv"

data = pd.read_csv(verified_images_csv, header=None)
selected_image_paths = data.iloc[:, 0]
selected_map_paths = data.iloc[:, 1]

# Get images and corresponding maps from the list of Verified IvyGap images
for i in range(len(selected_image_paths)):
    get_images_from_path(source_images_folder, selected_image_paths[i], images_folder)
    get_maps_from_path(source_maps_folder, selected_map_paths[i], maps_folder)

print(f"Total images copied: {len(os.listdir(images_folder))}")
print(f"Total maps copied: {len(os.listdir(maps_folder))}")

# Randomly select 120 image maps for acquiring normalization parameters
randomly_select_maps(csv_path, maps_folder, norm_maps_folder)

# Get their corresponding images
get_images_from_selected_maps(norm_maps_folder, images_folder, norm_images_folder)

norm_maps_folder = './data/for_normalization/Image_Maps'
list_norm_maps = os.listdir(norm_maps_folder)
csv_path = "data/Csv_Files/VerifiedIvyGap858ImageCohort.csv"
maps_folder = "./data/Image_Maps"

data = pd.read_csv(csv_path, header=None)
column = data.iloc[:,0]
all_map_names = []
for row in column:
    map_name = row.split("/")[-1]
    all_map_names.append(map_name)
for map_name in list_norm_maps:
    if map_name in all_map_names:
        all_map_names.remove(map_name)
print(f"Total (858-120): {len(all_map_names)}") # 738

maps_folder = "./data/Image_Maps"
available_maps = os.listdir(maps_folder) #1000

common_maps = list(set(all_map_names).intersection(available_maps))

print(f"Total maps to choose from: {len(common_maps)}")

random_maps = random.sample(common_maps, 5)
for map in random_maps:
    shutil.copy(os.path.join(maps_folder, map), "data/for_comparison/Image_Maps")

get_images_from_selected_maps("data/for_comparison/Image_Maps", "data/Images", "data/for_comparison/Images")