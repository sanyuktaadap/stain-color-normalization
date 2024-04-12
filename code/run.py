import os
import subprocess

# Images, image maps, tissue to exclude, and output dataframe name
image_array = os.listdir("./data/for_normalization/images")
image_map_array = os.listdir("./data/for_normalization/image_maps/")
# excluding_labels = ["", "", "Infiltrating Tumor", "Infiltrating Tumor"]
excluding_labels = [""] * 120
# output_dataframe_name = ["Dataframe_266290664", "Dataframe_268005945", "Dataframe_292324603", "Dataframe_292324711"]
# output_dataframe_name = [f'Dataframe_{image_array[0]}', f'Dataframe_{image_array[1]}', f'Dataframe_{image_array[2]}', f'Dataframe_{image_array[3]}']
output_dataframe_name = [f'Dataframe_{image}' for image in image_array[:]]

# Other
training_time = 10
knuth_bin_size = 4096

# Create Directories
directories_to_create = [
    "./results/Images_Histograms_DataFrames",
    "./results/Images_Stain_Stats_DataFrames",
    "./results/Normalization_Parameters",
    "./results/Normalized_Images",
    f"./results/Normalization_Parameters/{len(image_array)}_Image_Cohort_Aggregated_Normalization_Parameters"
]

for directory in directories_to_create:
    os.makedirs(directory, exist_ok=True)

# File Paths
python_scripts_directory = "./code/src/"
images_directory = "./data/for_normalization/images/"
image_maps_directory = "./data/for_normalization/image_maps/"
gray_level_labels_directory = "./data/Csv_Files/"
output_files = "./results/"

# 1. Calculate stain vectors and histogram for each image and store info in a dataframe
for i, image in enumerate(image_array):
    print("Generate pandas dataframes containing stain vectors and optical density for each cohort image")
    # Call another Python script from within the current Python script
    subprocess.run([
        "python", python_scripts_directory + "1-Produce_Image_Stain_Vectors_and_Optical_Density.py",
        # Input parameters required by the script:
        "--Slide_Image", images_directory + image,
        "--Label_Map_Image", image_maps_directory + image_map_array[i],
        "--Gray_Level_To_Label_Legend", gray_level_labels_directory + "LV_Gray_Level_to_Label.csv",
        "--Output_Dataframe_File", output_files + output_dataframe_name[i],
        "--Excluding_Labels", excluding_labels[i],
        "--Bin_Size", str(knuth_bin_size),
        "--Stain_Vector_Training", str(training_time)
    ])


# 2. Aggregate stain vectors and histogram from four images in step 1
print("Aggregate stain vectors and histograms")
subprocess.run([
    # Call another Python script from within the current Python script
    "python", python_scripts_directory + "2-Aggregate_Stain_Vectors_and_Histograms.py",
    "--Histogram_Dataframe_Directory", output_files + "Images_Histograms_DataFrames",
    "--Stain_Vector_Dataframe_Directory", output_files + "Images_Stain_Stats_DataFrames",
    "--Output_Directory", output_files + "Normalization_Parameters",
    "--Number_of_Images", str(len(image_array))
])

# # 3. Normalize each image using aggregated stain vectors and histogram in step 2
# for i, image in enumerate(image_array):
#     print("Normalize image using aggregated parameters")
#     # Call another Python script from within the current Python script
#     subprocess.run([
#         "python", python_scripts_directory + "Normalize_Image.py",
#         "--Image_To_Normalize", images_directory + image,
#         "--Normalizing_Histogram", output_files + f"Normalization_Parameters/{len(image_array)}_Image_Cohort_Aggregated_Normalization_Parameters/{len(image_array)}ImageCohortHistograms.npy",
#         "--Normalizing_Stain_Vectors", output_files + f"Normalization_Parameters/{len(image_array)}_Image_Cohort_Aggregated_Normalization_Parameters/{len(image_array)}ImageCohortStainVectors.npy",
#         "--Output_Directory", output_files + "Normalized_Images",
#         "--Stain_Vector_Training", str(training_time)
#     ])