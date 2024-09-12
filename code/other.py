from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# Load the images

# def calc_pi_vals_per_patch(image_path, patch_size, results_folder):
#     image_name = image_path.split("/")[-1].split(".")[0]
#     image = Image.open(image_path)

#     # Convert image to numpy array
#     image_array = np.array(image)

#     # Get the dimensions of the image
#     height, width, _ = image_array.shape

#     # Calculate the number of patches
#     num_patches_y = height // patch_size
#     num_patches_x = width // patch_size

#     # Initialize list to store the average RGB values
#     average_colors = []

#     # Loop through the image and extract patches
#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             # Define the patch
#             patch = image_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
#             # Calculate the average RGB values
#             avg_color = patch.mean(axis=(0, 1))
#             # Append the values to the list
#             average_colors.append(avg_color)

#     # Convert the list to a DataFrame
#     average_colors_df = pd.DataFrame(average_colors, columns=['R', 'G', 'B'])
#     # Save the DataFrame to a CSV file
#     average_colors_df.to_csv(f'{results_folder}/colors_per_patch_{image_name}_{patch_size}.csv', index=False)

# images = ["266290168.jpg", "266291277.jpg", "268005777.jpg", "286689393.jpg",
#           "286689429.jpg", "286689630.jpg", "286689702.jpg", "292342922.jpg",
#           "292343024.jpg", "294218229.jpg", "294218391.jpg", "294218963.jpg",
#           "300631399.jpg", "309714931.jpg", "311174553.jpg", "311174970.jpg"]

# print("Calculating pixels")
# for image in tqdm(images):
#     calc_pi_vals_per_patch(image_path=f"data/for_normalization/Images/{image}", patch_size=128, results_folder="results/temp_results")

files = os.listdir("results/temp_results")

dfs = []
for file in files:
    if ".csv" in file:
        dfs.append(file)

colors = ["crimson", 'palevioletred', 'mediumorchid', 'mediumpurple',
          'mediumblue', 'cornflowerblue', "royalblue", 'steelblue',
          'dodgerblue', 'deepskyblue', 'mediumseagreen', 'limegreen',
          'red', 'indianred', 'orange', 'gold']

print("Plotting graphs")
for idx, df in tqdm(enumerate(dfs)):
    data = pd.read_csv(f"results/temp_results/{df}")
    # gs_col1 = []
    gs_col2 = []
    # gs_col3 = []
    # gs_col4 = []

    for i in range(len(data)):
        r = data.iloc[i,0]
        g = data.iloc[i,1]
        b = data.iloc[i,2]
        # gs1 = (r + g + b) // 3
        # gs_col1.append(gs1)

        r2 = data.iloc[i,0] * 0.48
        g2 = data.iloc[i,1] * 0.32
        b2 = data.iloc[i,2] * 0.20
        gs2 = int(r2 + g2 + b2)
        gs_col2.append(gs2)

        # gs3 = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # gs_col3.append(gs3)

        # gs4 = (max(r, g, b) + min(r, g, b)) / 2
        # gs_col4.append(gs4)

    plot_name = df.split("patch_")[-1].split(".")[0]

    # plt.figure(figsize=(10, 10))
    # plt.hist(gs_col1, bins=250, edgecolor='red')
    # plt.axvline(x = 25)
    # plt.axvline(x = 90)
    # plt.axvline(x = 135)
    # plt.axvline(x = 160)
    # plt.axvline(x = 190)
    # plt.axvline(x = 225)
    # plt.axvline(x = 240)
    # plt.axvline(x = 255)
    # plt.title('Histogram (Mean)')
    # plt.xlabel('Pixel Values')
    # plt.ylabel('Frequency')
    # results_folder1 = "results/pi_per_patch/mean_colors"
    # plt.savefig(f'{results_folder1}/{plot_name}_mean.png')
    # plt.close()

    plt.figure(figsize=(10, 10))
    plt.hist(gs_col2, bins=200, color=colors[idx])
    plt.axvline(x = 25, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 90, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 135, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 160, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 190, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 225, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 240, c='black', linestyle = ':', linewidth = 3.0)
    plt.axvline(x = 255, c='black', linestyle = ':', linewidth = 3.0)
    plt.title('Histogram (Weighted)')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    results_folder2 = "results/temp_results/plots"
    plt.savefig(f'{results_folder2}/{plot_name}_weighted.png')
    plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.hist(gs_col3, bins=250, edgecolor='blue')
    # plt.axvline(x = 25)
    # plt.axvline(x = 90)
    # plt.axvline(x = 135)
    # plt.axvline(x = 160)
    # plt.axvline(x = 190)
    # plt.axvline(x = 225)
    # plt.axvline(x = 240)
    # plt.axvline(x = 255)
    # plt.title('Histogram (Luminance)')
    # plt.xlabel('Pixel Values')
    # plt.ylabel('Frequency')
    # results_folder3 = "results/pi_per_patch/luminance_colors"
    # plt.savefig(f'{results_folder3}/{plot_name}_luminance.png')
    # plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.hist(gs_col4, bins=250, edgecolor='yellow')
    # plt.axvline(x = 25)
    # plt.axvline(x = 90)
    # plt.axvline(x = 135)
    # plt.axvline(x = 160)
    # plt.axvline(x = 190)
    # plt.axvline(x = 225)
    # plt.axvline(x = 240)
    # plt.axvline(x = 255)
    # plt.title('Histogram (Desaturation)')
    # plt.xlabel('Pixel Values')
    # plt.ylabel('Frequency')
    # results_folder4 = "results/pi_per_patch/desaturation_colors"
    # plt.savefig(f'{results_folder4}/{plot_name}_desaturation.png')
    # plt.close()


# file = os.listdir("results/pi_per_patch")[1]
# data = pd.read_csv(f"results/pi_per_patch/{file}")
# gs_col1 = []

# for i in range(len(data)):
#     r = data.iloc[i,0]
#     g = data.iloc[i,1]
#     b = data.iloc[i,2]
#     gs1 = (r + g + b) // 3
#     gs_col1.append(gs1)

# plt.figure(figsize=(10, 10))
# plt.hist(gs_col1, bins=250, edgecolor='red')
# plt.axvline(x = 25, c='c', linestyle = '-.', linewidth = 3.0)
# plt.axvline(x = 90)
# plt.axvline(x = 135)
# plt.axvline(x = 160)
# plt.axvline(x = 190)
# plt.axvline(x = 225)
# plt.axvline(x = 240)
# plt.axvline(x = 255)
# plt.title('Histogram (Mean)')
# plt.xlabel('Pixel Values')
# plt.ylabel('Frequency')
# plt.show()