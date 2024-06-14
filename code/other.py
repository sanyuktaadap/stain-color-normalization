from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

Image.MAX_IMAGE_PIXELS = None

# Load the images

def calc_pi_vals_per_patch(image_path, patch_size, results_folder):
    image_name = image_path.split("/")[-1].split(".")[0]
    image = Image.open(image_path)

    # Convert image to numpy array
    image_array = np.array(image)

    # Get the dimensions of the image
    height, width, _ = image_array.shape

    # Calculate the number of patches
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Initialize list to store the average RGB values
    average_colors = []

    # Loop through the image and extract patches
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Define the patch
            patch = image_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            # Calculate the average RGB values
            avg_color = patch.mean(axis=(0, 1))
            # Append the values to the list
            average_colors.append(avg_color)

    # Convert the list to a DataFrame
    average_colors_df = pd.DataFrame(average_colors, columns=['R', 'G', 'B'])

    # Plot the average RGB values
    plt.figure(figsize=(20, 10))
    plt.plot(average_colors_df['R'], color='red', label='Red Channel')
    plt.plot(average_colors_df['G'], color='green', label='Green Channel')
    plt.plot(average_colors_df['B'], color='blue', label='Blue Channel')
    plt.xlabel(f'Patches ({patch_size})')
    plt.ylabel('Average Pixel Value')
    plt.title('Average RGB Values per Patch')
    plt.legend()
    # plt.show()
    plt.savefig(f'{results_folder}/{image_name}_{patch_size}.png')

    # Save the DataFrame to a CSV file if needed
    average_colors_df.to_csv(f'{results_folder}/colors_per_patch_{image_name}_{patch_size}.csv', index=False)

dfs = []
results_folder = "results/pi_per_patch"
res_files = os.listdir(results_folder)
for file in res_files:
    if ".csv" in file:
        dfs.append(file)

for df in dfs:
    results_folder = "results/pi_per_patch"
    image_name = df.split("_")[3]
    patch_size = df.split("_")[-1].split(".")[0]
    plot_name = f"{image_name}_{patch_size}.png"
    name = f"{results_folder}/{df}"
    df = pd.read_csv(name)
    df = df.astype(int)

    # Plot histograms for R, G, and B channels
    plt.figure(figsize=(10, 6))

    plt.hist(df['R'], bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.hist(df['G'], bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(df['B'], bins=256, color='blue', alpha=0.5, label='Blue Channel')

    plt.title('Histogram of RGB Intensity Values')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    # Adjust x-axis limits based on the combined min and max values of all channels
    combined_min = min(df['R'].min(), df['G'].min(), df['B'].min())
    combined_max = max(df['R'].max(), df['G'].max(), df['B'].max())
    xmin = []
    plt.xlim([combined_min, combined_max])

    # Show the plot
    plt.tight_layout()
    results_folder = "results/temp_results"
    plt.savefig(f'{results_folder}/{plot_name}')