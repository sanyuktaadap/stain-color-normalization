from setup_utils import *
import pandas as pd

verified_images_csv = "./data/Csv_Files/VerifiedIvyGap2176ImageCohortListWithExclusions.csv"
source_images_folder = "./data/RAW"
images_folder = "./data/Images"
source_maps_folder = "./data/label_maps/label_maps"
maps_folder = "./data/Image_Maps"
norm_images_folder = './data/for_normalization/Images'
norm_maps_folder = "./data/for_normalization/Image_Maps"
num_images = 120

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
randomly_select_images(maps_folder, num_images, norm_maps_folder)
# Get their corresponding images
get_images_from_selected_maps(norm_maps_folder, images_folder, norm_images_folder)