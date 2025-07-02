from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
from tqdm import tqdm
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_pixel_counts_from_csv(csv_path, output_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure correct column names
    df.columns = df.columns.str.strip()
    if 'Region Label' not in df or 'Pixel Count' not in df:
        raise ValueError("CSV must contain 'Region Label' and 'Pixel Count' columns.")

    # Exclude label 0 (background)
    df = df[df['Region Label'] != 0]

    labels = df['Region Label'].tolist()
    counts = df['Pixel Count'].tolist()

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, counts, color='mediumpurple')

    plt.xlabel("Region Label", fontsize=17)
    plt.ylabel("Total Pixel Count", fontsize=17)
    plt.title("Aggregated Pixel Count per Region", fontsize=17)
    plt.xticks(ticks=labels, labels=labels, ha='right', fontsize=17)

    # Use scientific notation on y-axis
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Plot saved at: {output_path}")


def load_pixel_count_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'Region Label' not in df or 'Pixel Count' not in df:
        raise ValueError("CSV must contain 'Region Label' and 'Pixel Count' columns.")
    df = df[df['Region Label'] != 0]
    return df

def plot_combined_pixel_counts(csv_path_1, csv_path_2, label_1, label_2, output_path):
    # Load both CSVs
    df1 = load_pixel_count_csv(csv_path_1)
    df2 = load_pixel_count_csv(csv_path_2)

    # Figure
    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # Font size (match x-axis label size)
    axis_fontsize = 17

    # Plot 1
    bars1 = axs[0].bar(df1['Region Label'], df1['Pixel Count'],
                       color='mediumseagreen', width=0.8)
    axs[0].set_xlabel("Region Label", fontsize=axis_fontsize)
    axs[0].set_ylabel("Total Pixel Count", fontsize=axis_fontsize)
    axs[0].set_title(label_1, fontsize=axis_fontsize)
    axs[0].set_xticks(df1['Region Label'])
    axs[0].tick_params(axis='x', labelsize=axis_fontsize)
    axs[0].tick_params(axis='y', labelsize=axis_fontsize)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Plot 2
    bars2 = axs[1].bar(df2['Region Label'], df2['Pixel Count'],
                       color='mediumseagreen', width=0.8)
    axs[1].set_xlabel("Region Label", fontsize=axis_fontsize)
    axs[1].set_title(label_2, fontsize=axis_fontsize)
    axs[1].set_xticks(df2['Region Label'])
    axs[1].tick_params(axis='x', labelsize=axis_fontsize)
    axs[1].tick_params(axis='y', labelsize=axis_fontsize)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Layout
    plt.suptitle("Aggregated Pixel Count per Region", fontsize=axis_fontsize + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Combined plot saved at: {output_path}")

# Usage
csv_path_1 = "results/plots/csv/pixel_count_stats_combined_Image_Maps.csv"
csv_path_2 = "results/plots/csv/pixel_count_stats_combined_KM_Masks.csv"
output_path = "results/plots/combined_pixel_count_stats.png"

plot_combined_pixel_counts(
    csv_path_1, csv_path_2,
    label_1="IvyGAP Masks",
    label_2="Unsupervised Masks",
    output_path=output_path
)

# Usage
csv_path = "results/plots/csv/pixel_count_stats_combined_Image_Maps.csv"
output_path = "results/plots/pixel_count_stats_combined_Image_Maps.png"
plot_pixel_counts_from_csv(csv_path, output_path)

csv_path = "results/plots/csv/pixel_count_stats_combined_KM_Masks.csv"
output_path = "results/plots/pixel_count_stats_combined_KM_Masks.png"
plot_pixel_counts_from_csv(csv_path, output_path)

# Path to images
maps = "Image_Maps"
images_fold = f"data/for_normalization/{maps}/*"
images_paths = glob.glob(images_fold)[:2]

# Initialize a global counter
global_pixel_counts = Counter()

# Loop through all images
for image_path in tqdm(images_paths):
    seg_map = Image.open(image_path).convert("L")  # Convert to grayscale
    seg_array = np.array(seg_map)
    pixel_counts = Counter(seg_array.flatten())
    global_pixel_counts.update(pixel_counts)  # Accumulate

# Sort and prepare data
sorted_labels = sorted(global_pixel_counts.keys())
counts = [global_pixel_counts[label] for label in sorted_labels]

# Save counts to CSV
df = pd.DataFrame({
    'Region Label': sorted_labels,
    'Pixel Count': counts
})
os.makedirs("results/csvs", exist_ok=True)
csv_path = f"results/plots/csv/pixel_count_stats_combined_{maps}.csv"
df.to_csv(csv_path, index=False)

# Plotting
plt.figure(figsize=(14, 6))
bars = plt.bar(sorted_labels, counts, color='mediumpurple')

# Add scientific notation count on top of each bar
for bar, count in zip(bars, counts):
    height = bar.get_height()
    sci_notation = f"{count:.1e}"
    plt.text(bar.get_x() + bar.get_width() / 2,
             height,
             sci_notation,
             ha='center',
             va='bottom',
             fontsize=17)

# Format axes
plt.xlabel("Region Label", fontsize=17)
plt.ylabel("Total Pixel Count", fontsize=17)
plt.title(f"Aggregated Pixel Count per Region", fontsize=17)
plt.xticks(ticks=sorted_labels, labels=sorted_labels, ha='right', fontsize=17)

# Use scientific notation on y-axis too
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Create results directory if it doesn't exist
os.makedirs("results/plots", exist_ok=True)
plt.savefig(f"results/plots/pixel_count_stats_combined_{maps}.png", dpi=300)